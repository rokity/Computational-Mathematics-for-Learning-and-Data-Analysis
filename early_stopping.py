import sys


class EarlyStopping:
    def __init__(self, monitor):
        """

        @param monitor: value used for the stop condition
        """
        self.monitor = monitor

    def get_monitor(self):
        """

        @return: monitor used
        """
        return self.monitor

    def early_stopping_check(self, history):
        """

        @param history: model history
        @return:
        """
        raise NotImplementedError


class GL(EarlyStopping):

    def __init__(self, monitor, alpha=4, patience=1, verbose=False):
        """

        @param monitor: value used for the stop condition
        @param alpha: absolute tolerance
        @param patience: number of epochs with no improvement after which training will be stopped.
        @param verbose: used for debug
        """
        super().__init__(monitor)
        self.alpha = alpha
        self.min_metric = sys.float_info.max
        self.patience = patience
        self.verbose = verbose

    def early_stopping_check(self, history):
        """

        @param history: model history
        @return: True if the condition is satisfied
                 False otherwise
        """
        metric = history[self.monitor][-1]
        gl = 100 * (metric / self.min_metric - 1)
        if self.verbose:
            print("gl: ", gl)
            print("patience (remaining): ", self.patience)

        if gl > self.alpha:
            self.patience -= 1
        if self.patience == 0:
            return True
        self.min_metric = min(self.min_metric, metric)
        return False


class PQ(EarlyStopping):

    def __init__(self, monitor, training_loss, alpha=4, k=5, verbose=False):
        """

        @param monitor: value used for the stop condition
        @param alpha: absolute tolerance
        @param k: training strip
        @param verbose: used for debug
        """
        super().__init__(monitor)
        self.alpha = alpha
        self.tr_loss = training_loss
        self.k = k
        self.init_PQ()
        self.min_metric = sys.float_info.max
        self.verbose = verbose

    def init_PQ(self):
        """

        initialize the training strip method
        """
        self.curr_k = 0
        self.min_tr_in_k = sys.float_info.max
        self.sum_tr_in_k = 0

    def early_stopping_check(self, history):
        """

        @param history: model history
        @return: True if the condition is satisfied
                 False otherwise
        """
        metric = history[self.monitor][-1]
        tr_loss = history[self.tr_loss][-1]
        gl = 100 * ((metric / self.min_metric) - 1)
        self.sum_tr_in_k += tr_loss
        if self.curr_k == self.k:
            pk = 1000 * (self.sum_tr_in_k / (self.k * self.min_tr_in_k) - 1)
            pq = gl / pk
            if self.verbose:
                print("gl: ", gl)
                print("pk: ", pk)
                print("pq: ", pq)

            if pq > self.alpha:
                return True
            else:
                self.init_PQ()

        self.min_metric = min(self.min_metric, metric)
        self.min_tr_in_k = min(self.min_tr_in_k, tr_loss)

        self.curr_k += 1
        return False