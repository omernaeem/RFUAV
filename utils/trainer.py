"""
定制训练器，由于使用默认的torch框架做。精度只有fp16
"""
class trainer():
   def __init__(self,
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs,
                num_class,
                save_path):
      self.model = model
      self.train_loader = train_loader
      self.val_loader = val_loader
      self.criterion = criterion
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.device = device
      self.num_epochs = num_epochs
      self.num_class = num_class
      self.save_path = save_path
      self.best_acc = 0
      self.best_loss = 1e6
      self.best_epoch = 0
      self.best_model = None
      
   def set_up(self):
      """
      set up the model
      """

   def train(self):
      """
      train model
      """
      
   def val(self):
      """
      val model
      """
      
   def save_model(self):
      """
      save model
      """