import os
import matplotlib.pyplot as plt
import itertools
import shutil
impot torch


def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, 'model.ckpt')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, "model_best.ckpt")
  

def save_cm_fig(cm, classes, normalize, title, dir):
    if normalize:
        cm = cm.astype('float')*100/cm.sum(axis=1)[:,None]
   
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks([])
#    plt.yticks([])
#    plt.xticks(tick_marks, classes, fontsize=)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt), fontsize=9,
                horizontalalignment='center',
                color='white' if cm[i,j] > threshold else 'black')
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('true', fontsize=18)
    plt.tight_layout()
    
    plt.savefig(f'{dir}/cm_{title}.jpg')
    return
