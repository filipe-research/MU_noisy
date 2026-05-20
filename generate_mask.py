import copy
import os
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import numpy as np


def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(args.save_dir, "with_{}.pt".format(i)))


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)

    model.cuda()

    def replace_loader_dataset(
        dataset_obj, batch_size_val=args.batch_size, seed_val=1, shuffle_val=True
    ):
        # utils.setup_seed(seed_val) # A seed é melhor configurada no worker_init_fn
        return torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=batch_size_val,
            num_workers=0, 
            pin_memory=True,
            shuffle=shuffle_val,
        )

    # --- Início da Lógica de Separação Forget/Retain ---
    # Fazemos cópias profundas do dataset original marcado para modificação
    forget_dataset_instance = copy.deepcopy(marked_loader.dataset)
    retain_dataset_instance = copy.deepcopy(marked_loader.dataset)

    # Determinar o nome do atributo dos rótulos (geralmente 'targets' ou 'labels')
    target_attr_name = ''
    if hasattr(forget_dataset_instance, 'targets'):
        target_attr_name = 'targets'
    elif hasattr(forget_dataset_instance, 'labels'):
        target_attr_name = 'labels'
    else:
        raise AttributeError(f"Dataset (tipo: {type(forget_dataset_instance)}) não possui atributo '.targets' ou '.labels'.")

    # Obter os rótulos originais do dataset marcado (alguns podem ser negativos)
    original_marked_targets = np.array(getattr(forget_dataset_instance, target_attr_name))

    # --- 1. Criar o Forget Set ---
    # Amostras para esquecer são aquelas com rótulos negativos
    is_forget_sample_mask = original_marked_targets < 0
    
    # Atualizar rótulos no forget_dataset_instance: reverter para original (positivo) e filtrar
    final_forget_targets = -original_marked_targets[is_forget_sample_mask] - 1
    setattr(forget_dataset_instance, target_attr_name, final_forget_targets)
    
    # Filtrar os dados das amostras (imagens)
    if hasattr(forget_dataset_instance, 'data') and getattr(forget_dataset_instance, 'data') is not None:
        # Para datasets como CIFAR10/100 que têm .data como NumPy array
        original_data_attr = np.array(getattr(forget_dataset_instance, 'data'))
        setattr(forget_dataset_instance, 'data', original_data_attr[is_forget_sample_mask])
        print(f"INFO: Forget set usou '.data' (shape: {forget_dataset_instance.data.shape if hasattr(forget_dataset_instance, 'data') else 'N/A'})")
    elif hasattr(forget_dataset_instance, 'image_paths') and getattr(forget_dataset_instance, 'image_paths') is not None:
        # Para seu Food101NDataset que usa .image_paths (lista de strings)
        original_paths_attr = np.array(getattr(forget_dataset_instance, 'image_paths'), dtype=object) # dtype=object para strings
        setattr(forget_dataset_instance, 'image_paths', original_paths_attr[is_forget_sample_mask].tolist())
        print(f"INFO: Forget set usou '.image_paths' (len: {len(forget_dataset_instance.image_paths) if hasattr(forget_dataset_instance, 'image_paths') else 'N/A'})")
    elif hasattr(forget_dataset_instance, 'imgs') and getattr(forget_dataset_instance, 'imgs') is not None:
        # Fallback para datasets que usam .imgs (geralmente lista de tuplas (caminho, rótulo) ou apenas caminhos)
        original_imgs_attr = list(getattr(forget_dataset_instance, 'imgs'))
        setattr(forget_dataset_instance, 'imgs', [img for i, img in enumerate(original_imgs_attr) if is_forget_sample_mask[i]])
        print(f"INFO: Forget set usou '.imgs' (len: {len(forget_dataset_instance.imgs) if hasattr(forget_dataset_instance, 'imgs') else 'N/A'})")
    else:
        print("AVISO: Não foi possível filtrar dados para forget_dataset. Atributo de dados (.data, .image_paths, .imgs) não encontrado ou é None.")

    # Opcional: Filtrar outros atributos se existirem e forem por amostra (ex: verification_labels)
    if hasattr(forget_dataset_instance, 'verification_labels') and getattr(forget_dataset_instance, 'verification_labels') is not None:
        original_vlabels = np.array(getattr(forget_dataset_instance, 'verification_labels'))
        setattr(forget_dataset_instance, 'verification_labels', original_vlabels[is_forget_sample_mask])
    
    forget_loader = replace_loader_dataset(forget_dataset_instance, seed_val=seed, shuffle_val=True)

    # --- 2. Criar o Retain Set ---
    # Amostras para reter são aquelas com rótulos NÃO negativos
    is_retain_sample_mask = original_marked_targets >= 0 # Usa os mesmos original_marked_targets

    # Atualizar rótulos no retain_dataset_instance: já estão corretos, apenas filtrar
    final_retain_targets = original_marked_targets[is_retain_sample_mask]
    setattr(retain_dataset_instance, target_attr_name, final_retain_targets)

    # Filtrar os dados das amostras (imagens)
    if hasattr(retain_dataset_instance, 'data') and getattr(retain_dataset_instance, 'data') is not None:
        original_data_attr_retain = np.array(getattr(retain_dataset_instance, 'data'))
        setattr(retain_dataset_instance, 'data', original_data_attr_retain[is_retain_sample_mask])
        print(f"INFO: Retain set usou '.data' (shape: {retain_dataset_instance.data.shape if hasattr(retain_dataset_instance, 'data') else 'N/A'})")
    elif hasattr(retain_dataset_instance, 'image_paths') and getattr(retain_dataset_instance, 'image_paths') is not None:
        original_paths_attr_retain = np.array(getattr(retain_dataset_instance, 'image_paths'), dtype=object)
        setattr(retain_dataset_instance, 'image_paths', original_paths_attr_retain[is_retain_sample_mask].tolist())
        print(f"INFO: Retain set usou '.image_paths' (len: {len(retain_dataset_instance.image_paths) if hasattr(retain_dataset_instance, 'image_paths') else 'N/A'})")
    elif hasattr(retain_dataset_instance, 'imgs') and getattr(retain_dataset_instance, 'imgs') is not None:
        original_imgs_attr_retain = list(getattr(retain_dataset_instance, 'imgs'))
        setattr(retain_dataset_instance, 'imgs', [img for i, img in enumerate(original_imgs_attr_retain) if is_retain_sample_mask[i]])
        print(f"INFO: Retain set usou '.imgs' (len: {len(retain_dataset_instance.imgs) if hasattr(retain_dataset_instance, 'imgs') else 'N/A'})")
    else:
        print("AVISO: Não foi possível filtrar dados para retain_dataset. Atributo de dados (.data, .image_paths, .imgs) não encontrado ou é None.")

    if hasattr(retain_dataset_instance, 'verification_labels') and getattr(retain_dataset_instance, 'verification_labels') is not None:
        original_vlabels_retain = np.array(getattr(retain_dataset_instance, 'verification_labels'))
        setattr(retain_dataset_instance, 'verification_labels', original_vlabels_retain[is_retain_sample_mask])

    retain_loader = replace_loader_dataset(retain_dataset_instance, seed_val=seed, shuffle_val=True)
    
    # --- Fim da Lógica de Separação ---

    # Validação do tamanho (Assertion)
    if train_loader_full and train_loader_full.dataset:
        # A asserção deve funcionar se __len__ dos datasets modificados estiver correto
        # (ou seja, se Food101NDataset.__len__ retorna len(self.targets) e self.targets é atualizado)
        if not (len(forget_dataset_instance) + len(retain_dataset_instance) == len(train_loader_full.dataset)):
            print(f"AVISO {args.dataset.upper()}: Asserção de tamanho falhou! "
                  f"Forget({len(forget_dataset_instance)}) + Retain({len(retain_dataset_instance)}) = {len(forget_dataset_instance) + len(retain_dataset_instance)}, "
                  f"Full({len(train_loader_full.dataset)})")
    else:
        print("AVISO: train_loader_full ou seu dataset é None. Não é possível verificar a asserção de tamanho.")

    print(f"Número de amostras no retain_dataset final: {len(retain_dataset_instance)}")
    print(f"Número de amostras no forget_dataset final: {len(forget_dataset_instance)}")
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    # Carregamento do modelo e chamada a save_gradient_ratio
    if args.resume:
        checkpoint_data = unlearn.load_unlearn_checkpoint(model, device, args) # Renomeado para evitar conflito
        if checkpoint_data is not None:
             model, evaluation_result = checkpoint_data # Desempacota se não for None
    else: # Sempre carrega o modelo base se não for resume
        checkpoint_loaded = torch.load(args.model_path, map_location=device) # Renomeado
        if "state_dict" in checkpoint_loaded.keys():
            state_dict_to_load = checkpoint_loaded["state_dict"]
        elif "model" in checkpoint_loaded.keys(): # Outro formato comum
            state_dict_to_load = checkpoint_loaded["model"]
        else:
            state_dict_to_load = checkpoint_loaded # Assume que o checkpoint é o state_dict diretamente

        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    save_gradient_ratio(unlearn_data_loaders, model, criterion, args)


if __name__ == "__main__":
    main()
