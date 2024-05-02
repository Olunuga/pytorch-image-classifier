class ClassifierData():
    def __init__(self, train, validation, test, class_index):
        self.train_dataloaders = train
        self.validation_dataloaders = validation
        self.test_dataloaders = test
        self.class_index = class_index