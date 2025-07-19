# ConvNeXt_Melanoma
Binary melanoma classification using ConvNeXt with 5-fold cross-validation and transfer learning

This project applies ConvNeXt (Tiny, Small, Base, Large) models to classify skin lesions as benign or malignant (melanoma). It uses transfer learning with ImageNet-pretrained weights and adds a dropout-regularized fully connected layer for 2-class classification.

- ConvNeXt variants
- Image preprocessing and data augmentation using `torchvision.transforms`
- Custom PyTorch dataset class for folder-based datasets
- 5-fold cross-validation using `sklearn.model_selection.KFold`
- Metrics: Accuracy, Precision, Recall, F1-score
