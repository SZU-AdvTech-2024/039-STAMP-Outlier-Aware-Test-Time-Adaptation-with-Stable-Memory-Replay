import logging

logger = logging.getLogger(__name__)
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.methods import setup_model
from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args
from src.utils.conf import cfg, load_cfg_fom_args
from src.data.data import load_ood_dataset_test

from src.methods.stamp import STAMP
from src.methods.tent import Tent
from src.methods.cotta import CoTTA
from src.models.Res import resnet18, ResNet

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9988))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

domain_order = 0
rand_select_size = 1500

# Ensure DEVICE is defined in cfg
if not hasattr(cfg, "DEVICE"):
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {cfg.DEVICE}")


def save_model(model, path):
    """
    Save the model's state_dict to the specified path.
    """
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved at {path}")

def extract_features(adapted_model, data_loader):
    """
    Extract features from the adapted model's underlying base model.
    """
    features = []
    labels = []

    # 如果是source
    if isinstance(adapted_model, ResNet):
        feature_layer = adapted_model.avg_pool
    # 如果是TTA方法
    else:
        feature_layer = adapted_model.model.avg_pool

    def hook(module, input, output):
        features.append(output.squeeze().detach().cpu().numpy()) 



    handle = feature_layer.register_forward_hook(hook)



    with torch.no_grad():
        print("列表长16 一个原图+15个增强视角 选择原始视角开始前向传播 捕获feature")
        for images, targets in data_loader:
         
            if isinstance(images, list):
                images = images[0].to(cfg.DEVICE) 
            else:
               
                images = images.to(cfg.DEVICE)

            if isinstance(adapted_model, ResNet ):
                _ = adapted_model(images)
            else:
                _ = adapted_model.model(images) 
            
            labels.extend(targets.cpu().numpy())

    handle.remove()

    print("特征提取结束")
    return np.concatenate(features, axis=0), np.array(labels)


def visualize_tsne(features, labels, domain_name):
    """
    Perform t-SNE dimensionality reduction and visualize the results.
    """
    perp = 6
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    features_2d = tsne.fit_transform(features)

    # Separate in-distribution (0-9) and out-of-distribution (255) samples
    in_dist_indices = labels != 255
    ood_indices = labels == 255

    plt.figure(figsize=(10, 8))

    # Plot in-distribution samples (0-9)
    scatter = plt.scatter(
        features_2d[in_dist_indices, 0], 
        features_2d[in_dist_indices, 1], 
        c=labels[in_dist_indices], 
        cmap='tab10', 
        alpha=0.7, 
        label="In-distribution"
    )

    # Plot out-of-distribution samples (255)
    if np.any(ood_indices):  # Check if there are OOD samples
        plt.scatter(
            features_2d[ood_indices, 0], 
            features_2d[ood_indices, 1], 
            c='black', 
            marker='x', 
            s=50, 
            label="Out-of-distribution"
        )

    # Add color bar for in-distribution classes
    cbar = plt.colorbar(scatter, ticks=range(10), label='In-distribution Class Labels')
    cbar.ax.set_yticklabels([str(i) for i in range(10)])  # Label the color bar with class numbers

    # Add title and legend
    plt.title("t-SNE Visualization of Test-Time Adaptation Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    global rand_select_size

    # plt.savefig(f"{domain_order}_perp{perp}_size{rand_select_size}_tsne_visualization_{domain_name}.png")
    plt.savefig(f"MODEL_{cfg.MODEL.ADAPTATION}_{domain_order}_perp{perp}_size{rand_select_size}_tsne_visualization_{domain_name}.png")
    print(f"{domain_order}_perp{perp}_size{rand_select_size}_Visualization saved as 'tsne_visualization_{domain_name}.png'")

def validation(cfg):
    model = setup_model(cfg)
    # get the test sequence containing the corruptions or domain names
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")


    severities = [cfg.CORRUPTION.SEVERITY[0]]

    accs = []
    aucs = []
    h_scores = []

    # Feature extraction and t-SNE visualization variables
    
    all_features = []
    all_labels = []

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_all):
        global domain_order # 声明要使用的全局变量
        domain_order += 1
        if cfg.MODEL.CONTINUAL == 'Fully':
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        elif cfg.MODEL.CONTINUAL == 'Continual':
            logger.warning("not resetting model")

        for severity in severities:
            testset, test_loader = load_ood_dataset_test(cfg.DATA_DIR, cfg.CORRUPTION.ID_DATASET,
                                                         cfg.CORRUPTION.OOD_DATASET, cfg.CORRUPTION.NUM_OOD_SAMPLES,
                                                         batch_size=cfg.TEST.BATCH_SIZE,
                                                         domain=domain_name, level=severity,
                                                         adaptation=cfg.MODEL.ADAPTATION,
                                                         workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                         ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                         # num_aug=cfg.TEST.N_AUGMENTATIONS if cfg.MODEL.ADAPTATION != 'stamp' else cfg.STAMP.NUM_AUG,
                                                         num_aug=cfg.TEST.N_AUGMENTATIONS if cfg.MODEL.ADAPTATION not in ['stamp','stamp_UA'] else cfg.STAMP.NUM_AUG)
                


            for epoch in range(cfg.TEST.EPOCH):
                acc, auc = get_accuracy(
                    model, data_loader=test_loader, cfg=cfg)
            h_score = 2 * acc * auc / (acc + auc)
            accs.append(acc)
            aucs.append(auc)
            h_scores.append(h_score)
            logger.info(
                f"{cfg.CORRUPTION.ID_DATASET} with {cfg.CORRUPTION.OOD_DATASET} [#samples={len(testset)}][{domain_name}]"
                f":acc: {acc:.2%}, auc: {auc:.2%}, h-score: {h_score:.2%}")

            #'''
            
            # Extract features for t-SNE visualization
            features, labels = extract_features(model, test_loader) # the model pass in is modified by TTA 
            all_features.append(features)
            all_labels.append(labels)
            #'''

        logger.info(f"mean acc: {np.mean(accs):.2%}, "
                    f"mean auc: {np.mean(aucs):.2%}, "
                    f"mean h-score: {np.mean(h_scores):.2%}")


        #'''
        # Perform t-SNE visualization
        print(f"Features shape: {all_features[0].shape}, Labels shape: {all_labels[0].shape}") 
        # Features shape: (225088, 512), Labels shape: (12500,)
        print(f"Domain: {domain_name}")


        
        np.random.seed(42) 
        global rand_select_size
        sampled_indices = np.random.choice(len(features), size=rand_select_size, replace=False)
        selected_features = features[sampled_indices]
        selected_labels = labels[sampled_indices]

        visualize_tsne(selected_features, selected_labels, domain_name)
        #'''

if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'evaluation_os'
    load_cfg_fom_args(args.cfg, args.output_dir) 
    merge_cfg_from_args(cfg, args) 
    cfg.CORRUPTION.SOURCE_DOMAIN = cfg.CORRUPTION.SOURCE_DOMAINS[0]
    logger.info(cfg)
    validation(cfg)
