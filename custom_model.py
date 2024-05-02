import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from torch import optim
import os

class FlowerClassifier():
    
    def __init__(self, supported_archs):
        self.dropout_probability = 0.2
        self.checkpoint_name = "checkpoint_flw.pth"
        self.supported_archs = supported_archs
        
    def train(self, classifier_data, epochs, lr, hidden_unit, arch, use_gpu, save_dir=None):
        self.classifier_data = classifier_data
        self.arch = arch
        self.arch_in_node = self.supported_archs[arch]
        self.classfier_arch = classifier_arch = {
            'hidden_unit': hidden_unit,
            'dropout_probablity':self.dropout_probability,
        }
        self.model = self._create_model(arch, self.classfier_arch)
        self.criterion = nn.NLLLoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu is True else "cpu")
        self.save_dir = save_dir
        
        optimizer = optim.Adam(params=self.model.classifier.parameters(), lr=lr)
        train_dataloaders = self.classifier_data.train_dataloaders
        validation_dataloaders = self.classifier_data.validation_dataloaders
        class_index = self.classifier_data.class_index
        
        accuracy = 0
        steps = 0
        print_every = 3
        
        self.model.to(self.device)
        
        print(f"\n... training {arch} architecture, starting with {epochs} epoch\n")
        for e in range(epochs):
            train_running_loss = 0
    
            for images, labels in train_dataloaders:
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                train_running_loss+= loss.item()

                loss.backward()
                optimizer.step()
                steps += 1

                if steps % print_every == 0:
                    test_running_loss = 0
                    accuracy = 0
                    self.model.eval()

                    for images, labels in validation_dataloaders:
                        images, labels = images.to(self.device), labels.to(self.device)

                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                        test_running_loss += loss.item()

                        ps = torch.exp(logits)
                        top_ps, top_class = ps.topk(1, dim=1)

                        equal = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equal.type(torch.FloatTensor))


                    print(f"Epoch: {e+1}/{epochs}... "
                          f"Training loss: {train_running_loss/steps:.3f}... "
                          f"Validation loss: {test_running_loss/len(validation_dataloaders):.3f}... "
                          f"validation accuracy: {accuracy/len(validation_dataloaders):.3f}..."
                         )

                    self.model.train()
                    train_running_loss = 0
        print(f"\n... Training completed with {(accuracy/len(validation_dataloaders)) * 100:.2f}%")
        
        save_path = self.checkpoint_name
        if self.save_dir is not None:
            save_path = saved_dir + "/" + save_path
     
        if self._save_checkpoint(save_path, self.model, optimizer, epochs, class_index, self.classfier_arch, self.arch, lr):
            print(f"... Model checkpoint saved in {save_path}..")
        else:
            print(f"... Error saving model checkpoint in {save_path}")
        
    
    def _create_model(self, arch, classifier_arch):
        model = self._create_model_for(arch)
        input_node_count = self.supported_archs[arch]
       
        for parameter in model.parameters():
            parameter.requires_grad = False
            
        classifier = self._create_classifier(input_node_count, classifier_arch)    
        model.classifier = classifier
        return model
    
    def _create_model_for(self, arch):
        model = None
        if arch == "densenet121":
            model = models.densenet121(pretrained=True)
        elif arch == "densenet161":
            model = models.densenet161(pretrained=True)
        elif arch == "densenet169":
            model = models.densenet169(pretrained=True)
        elif arch == "densenet201":
            model = models.densenet201(pretrained=True)
        elif arch == "vgg11":
            model = models.vgg11(pretrained=True)
        elif arch == "vgg13":
            model = models.vgg13(pretrained=True)
        elif arch == "vgg16":
            model = models.vgg16(pretrained=True)
        elif arch == "vgg19":
            model = models.vgg19(pretrained=True)
        elif arch == "alexnet":
            model = models.alexnet(pretrained=True)
        
        return model
            
    def predict(self, image, checkpoint_path, use_gpu, topk= 3, category_names= None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu is True else "cpu")
        print(f"Predicting on {self.device}: use_gpu is {use_gpu} and cuda available is {torch.cuda.is_available()}")
   
        # 1 - Load model from checkpoint
        model, checkpoint = self._load_inference_model(checkpoint_path)
        
        # 2 - Make inference with option to infer on GPU
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            logit = model(image)
            ps = torch.exp(logit)
            top_ps, top_class = ps.topk(topk, dim=1)

            top_ps = top_ps.to('cpu')
            top_class = top_class.to('cpu')

            top_ps = top_ps[0].numpy()
            top_class = top_class[0].numpy()

            class_to_idx = checkpoint['class_index']
            top_class = [f'{class_to_idx[c]}' for c in top_class]

        prediction = {}
        for index in range(0, len(top_class)):
            if category_names is not None:
                prediction[category_names[f"{top_class[index]}"]] = f"{top_ps[index] * 100 : .2f}%"
            else:
                prediction[f"{top_class[index]}"] = f"{top_ps[index] * 100 : .2f}%"
               
        print(prediction)
        
        
    def _create_classifier(self, input_node_count, classifier_arch):
        classifier = nn.Sequential(
            nn.Linear(input_node_count, classifier_arch['hidden_unit']),
            nn.ReLU(),
            nn.Dropout(p=classifier_arch['dropout_probablity']),
            nn.Linear(classifier_arch['hidden_unit'], 102),
            nn.LogSoftmax(dim=1))
        return classifier
    
    def _save_checkpoint(self, path, model, optimizer, epochs, classes_idx, classifier_architecture, arch, lr):
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        checkpoint = {
            'base_arch': arch,
            'classifier_arch': classifier_architecture,
            'epochs': epochs,
            'state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'class_index': classes_idx,
            "learning_rate": lr
        }
        
        torch.save(checkpoint, path)
        if os.path.exists(path):
            return True
        else:
            return False
    
    def _load_model_from(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = self._create_model(checkpoint['arch'], checkpoint['classifier_arch'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        return model, checkpoint
    
    def _load_inference_model(self, checkpoint_path):
        model, checkpoint = self._load_model_from(checkpoint_path)
        return model, checkpoint
        
    def _load_train_model(self, checkpoint_path):
        model, checkpoint = self._load_model_from(checkpoint_path)
        optimizer = optim.Adam(params=model.classifier.parameters(), lr=checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer