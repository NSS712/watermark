import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def draw(f1,f2,f3,f4, image, image_watermark, image_rotate, image_crop, image_compressed):
    (fpr1, tpr1, auc1), (fpr2, tpr2, auc2), (fpr3, tpr3, auc3), (fpr4, tpr4, auc4) = f1, f2, f3, f4
    # 创建子图
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    
    axes[0][0].imshow(Image.fromarray(image.transpose(1,2,0).astype(np.uint8)))
    axes[0][0].axis('off')
    axes[0][0].set_title(f"origin")
    axes[0][1].imshow(Image.fromarray(image_watermark.transpose(1,2,0).astype(np.uint8)))
    axes[0][1].axis('off')
    axes[0][1].set_title(f"water_mark")
    axes[0][2].imshow(Image.fromarray(image_rotate.transpose(1,2,0).astype(np.uint8)))
    axes[0][2].axis('off')
    axes[0][2].set_title(f"rotate")
    axes[0][3].imshow(Image.fromarray(image_crop.transpose(1,2,0).astype(np.uint8)))
    axes[0][3].axis('off')
    axes[0][3].set_title(f"crop")
    axes[0][4].imshow(Image.fromarray(image_compressed.transpose(1,2,0).astype(np.uint8)))
    axes[0][4].axis('off')
    axes[0][4].set_title(f"compressed")
    
    # 第一个子图
    axes[1][0].axis('off')
    axes[1][1].plot(fpr1, tpr1, label=f'ROC Curve (AUC = {auc1:.2f})', color='blue')
    axes[1][1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    axes[1][1].set_title('ROC Curve 1')
    axes[1][1].set_xlabel('False Positive Rate')
    axes[1][1].set_ylabel('True Positive Rate')
    axes[1][1].legend()

    # 第二个子图
    axes[1][2].plot(fpr2, tpr2, label=f'ROC Curve (AUC = {auc2:.2f})', color='green')
    axes[1][2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    axes[1][2].set_title('ROC Curve 2')
    axes[1][2].set_xlabel('False Positive Rate')
    axes[1][2].legend()

    # 第三个子图
    axes[1][3].plot(fpr3, tpr3, label=f'ROC Curve (AUC = {auc3:.2f})', color='red')
    axes[1][3].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    axes[1][3].set_title('ROC Curve 3')
    axes[1][3].set_xlabel('False Positive Rate')
    axes[1][3].legend()
    
    # 第四个子图
    axes[1][4].plot(fpr4, tpr4, label=f'ROC Curve (AUC = {auc4:.2f})', color='red')
    axes[1][4].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    axes[1][4].set_title('ROC Curve 4')
    axes[1][4].set_xlabel('False Positive Rate')
    axes[1][4].legend()    

    # 调整布局
    plt.tight_layout()
    plt.show()
    a = 1