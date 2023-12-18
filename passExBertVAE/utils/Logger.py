import time
import os
import sys


class Logger(object):

    def __init__(self, dataset=None, stream=sys.stdout, log_saved_path = None, running_operation=None):
        filename = log_saved_path
        if log_saved_path is None:
            output_dir = "log"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if dataset is not None:
                if running_operation is not None:
                    log_name = '{}_{}_{}.log'.format(dataset, running_operation, time.strftime('%Y-%m-%d-%H-%M'))
                else:
                    log_name = '{}_{}.log'.format(dataset, time.strftime('%Y-%m-%d-%H-%M'))
            else:
                if running_operation is not None:
                    log_name = '{}_{}.log'.format(running_operation, time.strftime('%Y-%m-%d-%H-%M'))
                else:
                    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
            filename = os.path.join(output_dir, log_name)
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
