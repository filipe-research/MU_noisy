import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate
import numpy as np


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
    seed = args.seed # Usado em replace_loader_dataset

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
        # utils.setup_seed(seed_val) # Seed é melhor configurada no worker_init_fn
        return torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=batch_size_val,
            num_workers=0, 
            pin_memory=True,
            shuffle=shuffle_val,
        )

    # --- Início da Lógica de Separação Forget/Retain (Adaptada de generate_mask.py e main_random.py) ---
    forget_dataset_instance = copy.deepcopy(marked_loader.dataset)
    retain_dataset_instance = copy.deepcopy(marked_loader.dataset)

    target_attr_name = ''
    if hasattr(forget_dataset_instance, 'targets'):
        target_attr_name = 'targets'
    elif hasattr(forget_dataset_instance, 'labels'):
        target_attr_name = 'labels'
    else:
        raise AttributeError(f"Dataset (tipo: {type(forget_dataset_instance)}) não possui atributo '.targets' ou '.labels'.")

    original_marked_targets = np.array(getattr(forget_dataset_instance, target_attr_name))

    # --- 1. Criar o Forget Set ---
    is_forget_sample_mask = original_marked_targets < 0
    
    final_forget_targets = -original_marked_targets[is_forget_sample_mask] - 1
    setattr(forget_dataset_instance, target_attr_name, final_forget_targets)
    
    if hasattr(forget_dataset_instance, 'data') and getattr(forget_dataset_instance, 'data') is not None:
        original_data_attr = np.array(getattr(forget_dataset_instance, 'data'))
        setattr(forget_dataset_instance, 'data', original_data_attr[is_forget_sample_mask])
    elif hasattr(forget_dataset_instance, 'image_paths') and getattr(forget_dataset_instance, 'image_paths') is not None:
        original_paths_attr = np.array(getattr(forget_dataset_instance, 'image_paths'), dtype=object)
        setattr(forget_dataset_instance, 'image_paths', original_paths_attr[is_forget_sample_mask].tolist())
    elif hasattr(forget_dataset_instance, 'imgs') and getattr(forget_dataset_instance, 'imgs') is not None:
        original_imgs_attr = list(getattr(forget_dataset_instance, 'imgs'))
        setattr(forget_dataset_instance, 'imgs', [img for i, img in enumerate(original_imgs_attr) if is_forget_sample_mask[i]])
    else:
        print("AVISO: Não foi possível filtrar dados para forget_dataset. Atributo de dados (.data, .image_paths, .imgs) não encontrado ou é None.")

    if hasattr(forget_dataset_instance, 'verification_labels') and getattr(forget_dataset_instance, 'verification_labels') is not None:
        original_vlabels = np.array(getattr(forget_dataset_instance, 'verification_labels'))
        setattr(forget_dataset_instance, 'verification_labels', original_vlabels[is_forget_sample_mask])
    
    forget_loader = replace_loader_dataset(forget_dataset_instance, seed_val=seed, shuffle_val=True)

    # --- 2. Criar o Retain Set ---
    is_retain_sample_mask = original_marked_targets >= 0

    final_retain_targets = original_marked_targets[is_retain_sample_mask]
    setattr(retain_dataset_instance, target_attr_name, final_retain_targets)

    if hasattr(retain_dataset_instance, 'data') and getattr(retain_dataset_instance, 'data') is not None:
        original_data_attr_retain = np.array(getattr(retain_dataset_instance, 'data'))
        setattr(retain_dataset_instance, 'data', original_data_attr_retain[is_retain_sample_mask])
    elif hasattr(retain_dataset_instance, 'image_paths') and getattr(retain_dataset_instance, 'image_paths') is not None:
        original_paths_attr_retain = np.array(getattr(retain_dataset_instance, 'image_paths'), dtype=object)
        setattr(retain_dataset_instance, 'image_paths', original_paths_attr_retain[is_retain_sample_mask].tolist())
    elif hasattr(retain_dataset_instance, 'imgs') and getattr(retain_dataset_instance, 'imgs') is not None:
        original_imgs_attr_retain = list(getattr(retain_dataset_instance, 'imgs'))
        setattr(retain_dataset_instance, 'imgs', [img for i, img in enumerate(original_imgs_attr_retain) if is_retain_sample_mask[i]])
    else:
        print("AVISO: Não foi possível filtrar dados para retain_dataset. Atributo de dados (.data, .image_paths, .imgs) não encontrado ou é None.")

    if hasattr(retain_dataset_instance, 'verification_labels') and getattr(retain_dataset_instance, 'verification_labels') is not None:
        original_vlabels_retain = np.array(getattr(retain_dataset_instance, 'verification_labels'))
        setattr(retain_dataset_instance, 'verification_labels', original_vlabels_retain[is_retain_sample_mask])

    retain_loader = replace_loader_dataset(retain_dataset_instance, seed_val=seed, shuffle_val=True)
    
    # --- Fim da Lógica de Separação ---

    # Renomeando variáveis para usar na assertiva e prints
    forget_dataset_final = forget_dataset_instance
    retain_dataset_final = retain_dataset_instance

    if train_loader_full and train_loader_full.dataset:
        if not (len(forget_dataset_final) + len(retain_dataset_final) == len(train_loader_full.dataset)):
            print(f"AVISO {args.dataset.upper()}: Asserção de tamanho falhou! "
                  f"Forget({len(forget_dataset_final)}) + Retain({len(retain_dataset_final)}) = {len(forget_dataset_final) + len(retain_dataset_final)}, "
                  f"Full({len(train_loader_full.dataset)})")
    else:
        print("AVISO: train_loader_full ou seu dataset é None. Não é possível verificar a asserção de tamanho.")

    print(f"Número de amostras no retain_dataset final: {len(retain_dataset_final)}")
    print(f"Número de amostras no forget_dataset final: {len(forget_dataset_final)}")
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint_data = unlearn.load_unlearn_checkpoint(model, device, args)
        if checkpoint_data is not None: # Verificar se o checkpoint foi carregado
            model, evaluation_result = checkpoint_data
    else:
        checkpoint_loaded = torch.load(args.model_path, map_location=device)
        state_dict_to_load = None
        if "state_dict" in checkpoint_loaded.keys():
            state_dict_to_load = checkpoint_loaded["state_dict"]
        elif "model" in checkpoint_loaded.keys(): # Outro formato comum
            state_dict_to_load = checkpoint_loaded["model"]
        else:
            state_dict_to_load = checkpoint_loaded # Assume que o checkpoint é o state_dict diretamente
        
        if state_dict_to_load: # Garantir que temos um state_dict para carregar
            new_state_dict = OrderedDict()
            for k, v in state_dict_to_load.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"AVISO: Não foi possível determinar o state_dict do checkpoint em {args.model_path}")


        # main_forget.py não usa args.mask_path, então o método de unlearning é chamado sem a máscara.
        # Se o seu método de unlearning específico precisar de uma máscara mesmo aqui, você precisará adicioná-la.
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args) # Chamada sem 'mask='
        unlearn.save_unlearn_checkpoint(model, None, args)


    if evaluation_result is None:
        evaluation_result = {}

    if "accuracy" not in evaluation_result: # Usando 'accuracy' como chave consistente
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            if loader is not None and loader.dataset is not None:
                # utils.dataset_convert_to_test(loader.dataset, args) # Comentado, aplicar se necessário
                val_acc = validate(loader, model, criterion, args)
                accuracy[name] = val_acc
                print(f"{name} acc: {val_acc:.2f}")
            else:
                print(f"AVISO: Loader '{name}' ou seu dataset é None. Pulando validação.")
                accuracy[name] = 0.0 
        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    
    # Cálculo de UA, RA, TA
    ua_accuracy = evaluation_result.get("accuracy", {}).get("forget", 0.0)
    UA = 100 - (ua_accuracy / 100 if isinstance(ua_accuracy, (int, float)) and ua_accuracy > 1 else ua_accuracy)
    print(f"UA (Unlearning Accuracy): {UA:.2f}% (baseado em forget acc: {ua_accuracy:.2f})")

    RA = evaluation_result.get("accuracy", {}).get("retain", 0.0)
    print(f"RA (Remaining Accuracy / Retain Accuracy): {RA:.2f}%")

    TA = evaluation_result.get("accuracy", {}).get("test", 0.0)
    print(f"TA (Test Accuracy): {TA:.2f}%")

    for deprecated_key in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated_key in evaluation_result:
            evaluation_result.pop(deprecated_key)

    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        if test_loader and test_loader.dataset and \
           forget_loader and forget_loader.dataset and \
           retain_loader and retain_loader.dataset:
            
            # utils.dataset_convert_to_test(retain_loader.dataset, args)
            # utils.dataset_convert_to_test(forget_loader.dataset, args)
            # utils.dataset_convert_to_test(test_loader.dataset, args)

            test_len_for_mia = len(test_loader.dataset)
            if len(retain_dataset_final) >= test_len_for_mia:
                shadow_train_subset_indices = list(range(test_len_for_mia))
                shadow_train_subset = torch.utils.data.Subset(retain_dataset_final, shadow_train_subset_indices)
                
                shadow_train_loader_mia = torch.utils.data.DataLoader(
                    shadow_train_subset, batch_size=args.batch_size, shuffle=False
                )

                evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                    shadow_train=shadow_train_loader_mia,
                    shadow_test=test_loader,
                    target_train=None, 
                    target_test=forget_loader,
                    model=model,
                )
                unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
            else:
                print(f"AVISO: Retain dataset (tamanho {len(retain_dataset_final)}) é menor que o test set (tamanho {test_len_for_mia}) para SVC_MIA.")
        else:
            print("AVISO: Um ou mais loaders/datasets necessários para SVC_MIA são None. Pulando cálculo.")

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
