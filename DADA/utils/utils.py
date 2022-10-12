import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def normalize_data(x):
   # step-1, fill nan as 0;
   #print("(max: %4f, min: %4f)" %(np.max(x), np.min(x)))
   x = np.nan_to_num(x)
   #print("(max: %4f, min: %4f)" %(np.max(x), np.min(x)))
   #print(np.min(x, axis=(0, 1)))
   #print(np.max(x, axis=(0, 1)))

   # ste-2: normalize data by column.
   #x_normed = (x - np.min(x, axis=(0,1), keepdims=True))/(np.max(x, axis=(0,1), keepdims=True) - np.min(x, axis=(0,1), keepdims=True) + 0.0000001)
   x_normed = (x - np.mean(x, axis=(0,1), keepdims=True))/(np.std(x, axis=(0,1), keepdims=True)  + 0.0000001)
   return x_normed


def Catergorical2OneHotCoding(a):
    # a: [1, 4, 0, 5, 2]
    # type: numpy array
    b = np.zeros((a.size, np.max(a) + 1))
    b[np.arange(a.size), a] = 1
    return b


def Logits2Binary(logits):
    pred = 1/( 1 + np.exp(-logits))
    return np.argmax(pred, axis=1)




def logits_2_multi_label(logits):
    #print(logits)
    pred = 1/( 1 + np.exp(-logits))
    threshold = 0.50
    #print(pred)
    pred[pred >= threshold] = 1;
    pred[pred < threshold] = 0;

    return pred


def print_genotype(logging, geno):
    for i, sub_policy in enumerate(geno):
        logging.info("%d: %s %f" %
                     (i, '\t'.join(["(%s %f %f)" % (x[0], x[1], x[2]) for x in sub_policy]), sub_policy[0][3]))
    geno_out = [[(x[0], x[1], x[2]) for x in sub_policy] for sub_policy in geno]
    #logging.info("genotype_%d: %s" % ( len(geno_out), str(geno_out) ))
    #logging.info("genotype_%d: %s" % ( len(geno_out[0:5]), str(geno_out[0:5]) ))
    #logging.info("genotype_%d: %s" % ( len(geno_out[0:10]), str(geno_out[0:10]) ))






def test():
    a = np.array([0, 1, 2, 3, 4, 5, 9])
    print(a)
    b = Catergorical2OneHotCoding(a)
    print(b)

    c = Logits2Binary(b)
    print(c)

if __name__ == "__main__":
    test()
    print("Everything passed")

