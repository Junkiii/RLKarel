import os
from datetime import datetime


class Logger():
    def __init__(self, logdir):
        print("Created new training logger!")
        logdir = logdir
        now = datetime.now()
        runname = "run_" + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S")

        self.path = os.path.join(logdir, runname)
        os.mkdir(self.path)

        print(">>>\tCreated new run: ", self.path)

        self.loss_file = open(os.path.join(self.path, "loss_log.txt"), "w")
        self.rewd_file = open(os.path.join(self.path, "rewd_log.txt"), "w")
        self.corr_file = open(os.path.join(self.path, "corr_log.txt"), "w")
        self.time_file = open(os.path.join(self.path, "time_log.txt"), "w")
        self.enr_time_file = open(os.path.join(self.path, "enr_time_log.txt"), "w")
        self.bwd_time_file = open(os.path.join(self.path, "bwd_time_log.txt"), "w")

        self.meta_file = open(os.path.join(self.path, "meta_log.txt"), "w")

        self.loss = list()
        self.rewd = list()
        self.corr = list()
        self.time = list()
        self.enr_time = list()
        self.bwd_time = list()

    def log_meta(self, batch_size, lr, num_epochs, optimizer, data, model):
        
        self.meta_file.write("Training log meta data:\n")
        self.meta_file.write(">>>   batch_size = " + str(batch_size) + "\n")
        self.meta_file.write(">>>   initial_lr = " + str(lr) + "\n")
        self.meta_file.write(">>>   num_epochs = " + str(num_epochs) + "\n")

        self.meta_file.write("\n")
        self.meta_file.write("Training data: \n")
        self.meta_file.write(data)
        self.meta_file.write("\n")

        self.meta_file.write("\n")
        self.meta_file.write("Optimizer: \n")
        self.meta_file.write(str(optimizer))
        self.meta_file.write("\n")
        
        self.meta_file.write("\n")
        self.meta_file.write("Model: \n")
        self.meta_file.write(str(model))
        self.meta_file.write("\n")
        
    def log_start(self):
        now = datetime.now()
        self.meta_file.write("Started training: " + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S"))
        self.meta_file.write("\n")

    def log_end(self, epoch):
        now = datetime.now()
        self.meta_file.write("Finished training: " + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S"))
        self.meta_file.write("\n")
        self.meta_file.write("Trained for " + str(epoch) + " epochs.")
        self.meta_file.write("\n")

    def log_all(self, iter, loss, rewd, corr, time, enr_time, bwd_time):
        print(f'Epoch {iter} \t||  Loss: {loss:.4f} \t|  Avg.Reward: {rewd:.3f} \t|  Correct: {corr:.3f}% \t|  Time: {time:.2f}s \t|  Enroll Time: {enr_time:.2f}% \t|  Backward Time: {bwd_time:.2f}%')
        self.log_loss(iter, loss)
        self.log_reward(iter, rewd)
        self.log_correct(iter, corr)
        self.log_time(iter, time)
        self.log_enroll_time(iter, enr_time)
        self.log_backward_time(iter, bwd_time)

    def log_loss(self, iter, val):
        self.loss.append((iter, val))

    def log_reward(self, iter, val):
        self.rewd.append((iter, val))
    
    def log_correct(self, iter, val):
        self.corr.append((iter, val))

    def log_time(self, iter, val):
        self.time.append((iter, val))

    def log_enroll_time(self, iter, val):
        self.enr_time.append((iter, val))

    def log_backward_time(self, iter, val):
        self.bwd_time.append((iter, val))

    def flush(self):
        for l in self.loss:
            self.loss_file.write(str(l) + '\n')
        for r in self.rewd:
            self.rewd_file.write(str(r) + '\n')
        for c in self.corr:
            self.corr_file.write(str(c) + '\n')
        for t in self.time:
            self.time_file.write(str(t) + '\n')
        for e in self.enr_time:
            self.enr_time_file.write(str(e) + '\n')
        for b in self.bwd_time:
            self.bwd_time_file.write(str(b) + '\n')

        self.loss.clear()
        self.rewd.clear()
        self.corr.clear()
        self.time.clear()
        self.enr_time.clear()
        self.bwd_time.clear()

    def close(self):
        self.flush()
        self.loss_file.close()
        self.meta_file.close()
        self.rewd_file.close()
        self.corr_file.close()
        self.time_file.close()
        self.enr_time_file.close()
        self.bwd_time_file.close()
        print(">>>\tLogger closed!")

    def get_run_dir(self):
        return self.path

    def is_checkpoint(self, iter):
        return (iter + 1) % 20 == 0


class NoneLogger(Logger):
    def __init__(self):
        print("No training logger active!")
    
    def log_start(self):
        pass

    def log_all(self, iter, loss, rewd, corr, time, enr_time, bwd_time):
        pass

    def log_end(self, epoch):
        pass

    def log_meta(self, batch_size, lr, num_epochs, optimizer, data, model):
        pass

    def flush(self):
        pass

    def close(self):
        pass
    