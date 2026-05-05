# metrics.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np
try:
    from . import backend as BE
except:
    import backend as BE


class Metrics:

    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.og_y_pred = y_pred
        self.one_hot_vector(pred=y_pred)
        self.report_flag = False

        if self.y_test.shape != self.y_pred.shape: raise ValueError("Prediction and Test arrays do not match in shape.")

        if self.y_test.shape[1] == 1:
            y_pred = (self.og_y_pred >= 0.5).astype(int)
            self.TP = BE.xp.sum((y_pred == 1) & (y_test == 1))
            self.FP = BE.xp.sum((y_pred == 1) & (y_test == 0))
            self.FN = BE.xp.sum((y_pred == 0) & (y_test == 1))
            self.TN = BE.xp.sum((y_pred == 0) & (y_test == 0))


    def one_hot_vector(self, pred):
        y_pred_clip = []
        for yi in pred:
            one_hot = BE.xp.zeros_like(yi)
            one_hot[BE.xp.argmax(yi)] = 1
            y_pred_clip.append(one_hot)
        self.y_pred = BE.xp.asarray(y_pred_clip)

    def confusion_matrix(self, graphical = False):

        if self.y_test.shape[1] == 1:
            self.conf_matrix = BE.xp.array([[self.TP, self.FN], [self.FP, self.TN]])

        else:
            self.conf_matrix = BE.xp.zeros((self.y_test.shape[1], self.y_test.shape[1]), dtype=int)
            for yt, yp in zip(self.y_test, self.y_pred):
                self.conf_matrix[BE.xp.argmax(yt), BE.xp.argmax(yp)] += 1

            self.TP = []
            self.FN = []
            self.FP = []
            self.TN = []
            
            for i in range(self.y_test.shape[1]):
                self.TP.append(self.conf_matrix[i, i])
                self.FN.append(BE.xp.sum(self.conf_matrix[i, :]) - self.TP[i])
                self.FP.append(BE.xp.sum(self.conf_matrix[:, i]) - self.TP[i])
                self.TN.append(BE.xp.sum(self.conf_matrix) - (self.TP[i] + self.FP[i] + self.FN[i]))
        
        if graphical:
            import matplotlib.pyplot as plt
            plt.imshow(self.conf_matrix, cmap="coolwarm")
            if self.y_test.shape[1] == 1:
                plt.xticks([0, 1], labels=["Positive", "Negative"])
                plt.yticks([0, 1], labels=["Positive", "Negative"])
            else:
                plt.xticks(BE.xp.arange(0, self.y_test.shape[1], step=1), labels=BE.xp.arange(1, self.y_test.shape[1]+1, step=1))
                plt.yticks(BE.xp.arange(0, self.y_test.shape[1], step=1), labels=BE.xp.arange(1, self.y_test.shape[1]+1, step=1))
            plt.ylabel("Actual")
            plt.xlabel("Prediction")
            plt.title("Heatmap of the confusion matrix")
            plt.colorbar()
            plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            for i in range(self.conf_matrix.shape[0]):
                for j in range(self.conf_matrix.shape[0]):
                    text = plt.text(j, i, self.conf_matrix[i, j],
                                ha="center", va="center", color="w")
            plt.show()

        return self.conf_matrix

    def accuracy(self):
        self.confusion_matrix()
        accuracy_list = []
        try:
            for i in range(self.y_test.shape[1]):
                accuracy_list.append((self.TP[i] + self.TN[i])/(self.TP[i] + self.TN[i] + self.FP[i] + self.FN[i] + 1e-8))
        except:
            accuracy_list.append((self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN + 1e-8))

        return accuracy_list

    def precision(self):
        self.confusion_matrix()
        precision_list = []
        try:
            for i in range(self.y_test.shape[1]):
                precision_list.append((self.TP[i])/(self.TP[i] + self.FP[i] + 1e-8))
        except:
            precision_list.append((self.TP)/(self.TP + self.FP + 1e-8))

        return precision_list

    def recall(self):
        self.confusion_matrix()
        recall_list = []
        try:
            for i in range(self.y_test.shape[1]):
                recall_list.append((self.TP[i])/(self.TP[i] + self.FN[i] + 1e-8))
        except:
            recall_list.append((self.TP)/(self.TP + self.FN + 1e-8))

        return recall_list
    
    def sensitivity(self):
        return self.recall()

    def F1score(self):
        p = self.precision()
        r = self.recall()
        F1_list = []
        try:
            for i in range(len(p)):
                F1_list.append(2*(p[i]*r[i])/(p[i]+r[i] + 1e-8))
        except:
            F1_list.append(2*(p[0]*r[0])/(p[0]+r[0] + 1e-8))
        return F1_list
    
    def ROC_for_class(self, k):

        y_true_k = self.y_test[:, k]        # 1 for class k, 0 otherwise
        y_prob_k = self.og_y_pred[:, k]        # predicted probability for class k
        
        #thrs = BE.xp.linspace(0, 1, 200)
        thrs = BE.xp.arange(0, 1, step=self.threshold_step)
        tpr = []
        fpr = []

        for t in thrs:
            y_pred_k = (y_prob_k >= t).astype(int)

            TP = BE.xp.sum((y_pred_k == 1) & (y_true_k == 1))
            FP = BE.xp.sum((y_pred_k == 1) & (y_true_k == 0))
            FN = BE.xp.sum((y_pred_k == 0) & (y_true_k == 1))
            TN = BE.xp.sum((y_pred_k == 0) & (y_true_k == 0))

            tpr.append(TP / (TP + FN + 1e-8))
            fpr.append(FP / (FP + TN + 1e-8))

        order = BE.xp.argsort(fpr)
        tpr = BE.xp.array(tpr)
        fpr = BE.xp.array(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        try:
            auc_k = BE.xp.trapz(tpr_sorted, fpr_sorted)
        except:
            auc_k = BE.xp.trapezoid(tpr_sorted, fpr_sorted)

        return fpr_sorted, tpr_sorted, auc_k
    
    def AUC(self, graphical = False, threshold_step = 5e-3):
        self.threshold_step = threshold_step
        if self.y_test.shape[1] > 1:
            roc_scores = []
            auc_scores = []
            for k in range(self.y_test.shape[1]):
                results = self.ROC_for_class(k)
                roc_scores.append([results[0], results[1]])
                auc_scores.append(results[2])

            macro_auc = BE.xp.mean(auc_scores)

            if graphical:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,6))
                for k in range(self.y_test.shape[1]):
                    plt.plot(roc_scores[k][0], roc_scores[k][1], label = f"Class {k+1}: AUC = {auc_scores[k]:.3g}")
                avg_roc = BE.xp.mean(roc_scores, axis=0)
                plt.plot(avg_roc[0], avg_roc[1], label = f"macro-average: AUC = {macro_auc:.3g}", linestyle = "--")
                plt.title("ROC curves for each class")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.legend(loc='lower right')
                plt.show()

            if self.report_flag:
                return macro_auc, roc_scores, auc_scores
            else:
                return macro_auc
        else:
            thrs = BE.xp.arange(0, 1, step=self.threshold_step)
            tpr = []
            fpr = []
            y_true = self.y_test
            for t in thrs:
                y_pred = (self.og_y_pred >= t).astype(int)

                TP = BE.xp.sum((y_pred == 1) & (y_true == 1))
                FP = BE.xp.sum((y_pred == 1) & (y_true == 0))
                FN = BE.xp.sum((y_pred == 0) & (y_true == 1))
                TN = BE.xp.sum((y_pred == 0) & (y_true == 0))

                tpr.append(TP / (TP + FN + 1e-8))
                fpr.append(FP / (FP + TN + 1e-8))

            order = BE.xp.argsort(fpr)
            tpr = BE.xp.array(tpr)
            fpr = BE.xp.array(fpr)
            fpr_sorted = fpr[order]
            tpr_sorted = tpr[order]
            try:
                auc = BE.xp.trapz(tpr_sorted, fpr_sorted)
            except:
                auc = BE.xp.trapezoid(tpr_sorted, fpr_sorted)

            if graphical:
                import matplotlib.pyplot as plt
                plt.plot(fpr_sorted, tpr_sorted)
                plt.title(f"ROC curve: AUC = {auc:.3g}")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.show()
        
            if self.report_flag:
                return auc, fpr_sorted, tpr_sorted
            else:
                return auc

    def report(self, graphical = True, threshold_step = 5e-3):
        import matplotlib.pyplot as plt

        self.report_flag = True
        conf_matrix = self.confusion_matrix()
        print("Confusion matrix:\n",conf_matrix)

        acc = self.accuracy()
        prec = self.precision()
        sens = self.sensitivity()
        F1 = self.F1score()

        num_classes = len(acc)
        classes_array = np.linspace(1, num_classes, num = num_classes, dtype=int)
        names_rep = np.array(["Class     ", "Accuracy ", "Precision", "Recall   ", "F1-score "])
        report_array = np.array([classes_array, acc, prec, sens, F1, ]).T
        #print(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {sens}\nF1-score: {F1}")  

        print()
        for i in range(5):
            if i == 0:
                print("Class        ","      ".join([f"{cl}" for cl in classes_array if cl < 10]),"".join([f"     {cl}" for cl in classes_array if cl >= 10]))
                print("----------")
                continue
            print(names_rep[i], "|", ", ".join([f"{item[i]:.3f}" for item in report_array]))
        print()

        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(1, 2, 1)

        if self.y_test.shape[1] > 1:

            auc, roc_curve, auc_curve = self.AUC(graphical=False, threshold_step=threshold_step)  
            print(f"AUC: {auc:.3f}")

            for k in range(self.y_test.shape[1]):
                ax1.plot(roc_curve[k][0], roc_curve[k][1], label = f"Class {k+1}: AUC = {auc_curve[k]:.3g}")
            avg_roc = BE.xp.mean(roc_curve, axis=0)
            ax1.plot(avg_roc[0], avg_roc[1], label = f"macro-average: AUC = {auc:.3g}", linestyle = "--")
            ax1.set_title("ROC curves for each class")
            ax1.set_xlabel("FPR")
            ax1.set_ylabel("TPR")
            ax1.legend(loc='lower right')

        else:
            auc, fpr_sorted, tpr_sorted = self.AUC(graphical=False, threshold_step=5e-3)  
            print(f"AUC: {auc:.3f}")

            ax1.plot(fpr_sorted, tpr_sorted)
            ax1.set_title(f"ROC curve: AUC = {auc:.3g}")
            ax1.set_xlabel("FPR")
            ax1.set_ylabel("TPR")

        ax2 = fig.add_subplot(1, 2, 2)

        ax2.imshow(conf_matrix, cmap="coolwarm")
        if self.y_test.shape[1] > 1:
            ax2.set_xticks(BE.xp.arange(0, self.y_test.shape[1], step=1), labels=BE.xp.arange(1, self.y_test.shape[1]+1, step=1))
            ax2.set_yticks(BE.xp.arange(0, self.y_test.shape[1], step=1), labels=BE.xp.arange(1, self.y_test.shape[1]+1, step=1))
        else:
            ax2.set_xticks([0, 1], labels=["Positive", "Negative"])
            ax2.set_yticks([0, 1], labels=["Positive", "Negative"])
        ax2.set_ylabel("Actual")
        ax2.set_xlabel("Prediction")
        ax2.set_title("Heatmap of the confusion matrix")
        ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[0]):
                ax2.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
        if graphical:
            plt.show()
        else:
            plt.close()
        
        self.report_flag = False

if __name__ == "__main__":
    classes = 1
    data = 1000
    logits = BE.xp.random.randn(data, classes)
    exp_logits = BE.xp.exp(logits)
    pred = exp_logits / BE.xp.sum(exp_logits, axis=1, keepdims=True)
    true = BE.xp.random.randn(data, classes)
    tmp = []
    if classes > 1:
        for yi in true:
            one_hot = BE.xp.zeros_like(yi)
            one_hot[BE.xp.argmax(yi)] = 1
            tmp.append(one_hot)
        true = BE.xp.asarray(tmp)
    else:
        pred = BE.xp.random.randn(data, classes)
        pred += abs(BE.xp.min(pred))
        pred /= BE.xp.max(pred)

        for yi in true:
            if yi >= 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        true = BE.xp.asarray(tmp).reshape(-1, 1)

    metrics = Metrics(y_test=true, y_pred=pred)
    metrics.report(graphical=True)

    auc = metrics.AUC(graphical=True, threshold_step=0.001)
    print("AUC: ", auc)
