from configparser import Interpolation
import copy
import random
from os import mkdir, path
from pathlib import Path
from pickle import dump, load
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from datetime import datetime

#https://sandipanweb.wordpress.com/2018/01/06/eigenfaces-and-a-simple-face-detector-with-pca-svd-in-python/


def loadImages(root):
    print("Loading Data...")
    DATA = [mpimg.imread(imgPath) for imgPath in list(map(str,Path(root).glob("*/*")))[:10000]]
    print("Done.")
    return DATA


def loadData(rawDataPath,cutoff):
    if not path.exists("./Preprocess"):
        print("No preprocessed data detected.\nProcessing raw data...")
        mkdir("./Preprocess")
        DATA = loadImages(rawDataPath)
        SHAPE = DATA[0].shape
        dump(SHAPE,open("./Preprocess/shape.p",'wb'))
        FLAT = np.matrix([np.reshape(img,(1,-1))[0] for img in DATA])
        U,S,Vh = np.linalg.svd(FLAT,full_matrices=False)
        dump((U,S,Vh),open("./Preprocess/svd.p",'wb'))
        save_truncated_svd(U,S,Vh,cutoff)
    else:
        SHAPE = load(open("./Preprocess/shape.p",'rb'))
        U,S,Vh = load(open("./Preprocess/svd_truncated_{}.p".format(cutoff),'rb'))

    return U,S,Vh,SHAPE

def save_truncated_svd(U,S,Vh,cutoff):
    dump((U[:,:cutoff], S[:cutoff], Vh[:cutoff,:]),open("./Preprocess/svd_truncated_{}.p".format(cutoff),'wb'))

# def unitTest_Reconstruct(FLAT,U,S,Vh,cutoff):
#     #reconstruct the original data set from the approximated data set and test for closeness
#     Smat = np.zeros((U.shape[1],Vh.shape[0]))
#     Smat[:S.shape[0],:S.shape[0]] = np.diag(S)
#     RECON = np.dot(U[:,:cutoff], np.dot(Smat[:cutoff,:cutoff],Vh[:cutoff,:]))
#     print("Reconstruction success?", np.allclose(FLAT, RECON))
#     img = np.reshape(np.array(RECON[0]),SHAPE)
#     plt.imshow(img)
#     plt.show()


# def unitTest_GenerateRandomFaces(U,S,Vh,SHAPE):
#     # generate 10 random faces
#     for _ in range(10):
#        rand = [random.uniform(-1,1)/50 for _ in range(U.shape[1])]
#        img = np.reshape(np.array(np.dot(rand, np.dot(np.diag(S), Vh))),SHAPE)
#        plt.imshow(img)
#        plt.show()


def unitTest_RSVs(S,Vh,SHAPE,save=False):
    # show 10 right singlular vectors (scaled up by S)
    for i,RSV in enumerate(np.dot(np.diag(S),Vh)[:10]):
        img = np.reshape(np.array(RSV),SHAPE)
        plt.imshow(img)
        if save:
            plt.savefig("UNIT_TEST_RSV_{}.png".format(i))
            plt.close()
        else:
            plt.show()


class Index:
    def __init__(self):
        self.choice = -1 #window closed by X in top right (default)

    def right(self, event):
        self.choice = 1
        # print("right")
        plt.close()

    def left(self, event):
        self.choice = 0
        # print("left")
        plt.close()

    def exit(self,event):
        self.choice = 2
        plt.close()

    def dump(self,event):
        self.choice = 3
        plt.close()


def clamp(val):
    print(val)
    return min(1,max(0,val))


 # ------------------------------------- evo
class Org:
    def __init__(self):
        global GENOME_LENGTH
        self.genome = [random.uniform(-1,1)/50 for _ in range(GENOME_LENGTH)]

    def make_mutated_copy(self):
        global GENOME_LENGTH,MUTATION_RATE
        child = Org()
        child.genome = copy.deepcopy(self.genome)
        for i in range(GENOME_LENGTH):
            if random.random() <= MUTATION_RATE:
                # PN = random.randint(0,1)
                #child.genome[i] += ((-1*PN)+(1-PN))/100 #if PN = 0, 1 ; if PN=1 , -1
                child.genome[i] += np.random.default_rng().normal(0.0,0.05)
        return child


def recombine(org1, org2):
    child = Org()
    #single cut crossover
    cut = random.randint(0,GENOME_LENGTH-1)
    child.genome = copy.deepcopy(org1.genome[:cut]) + copy.deepcopy(org2.genome[cut:])

    #all-sites crossover
    # child.genome = []
    # for site in range(GENOME_LENGTH):
    #     if random.random() <= 0.5:
    #         child.genome.append(org1.genome[site])
    #     else:
    #         child.genome.append(org2.genome[site])

    
    return child


def imageTournament(org1,org2):
    global S, Vh, SHAPE, logdata, aveFace
    img1 = np.reshape(np.array(np.clip(np.dot(org1.genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE) #TODO: the need to manually truncate is obsolete
    img2 = np.reshape(np.array(np.clip(np.dot(org2.genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
    imgAve = np.reshape(np.array(np.clip(np.dot(aveFace, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)

    f, axarr = plt.subplots(2,2)
    f.set_size_inches((10, 10))

    axarr[0][0].set_title("Rate of Change")
    axarr[0][0].plot(logdata)
    axarr[0][1].set_title("Population Average")
    axarr[0][1].imshow(imgAve, interpolation="quadric")
    axarr[1][0].set_title("Left Choice")
    axarr[1][0].imshow(img1, interpolation="quadric")
    axarr[1][1].set_title("Right Choice")
    axarr[1][1].imshow(img2, interpolation="quadric")
    
    callback = Index()
    axprev = plt.axes([0.7, 0.005, 0.1, 0.075]) #left start, bottom start, width, height
    axnext = plt.axes([0.81, 0.005, 0.1, 0.075])
    axexit = plt.axes([0.1, 0.005, 0.11, 0.075])
    axdump = plt.axes([0.22, 0.005, 0.2, 0.075])
    bnext = Button(axnext, 'Right')
    bnext.on_clicked(callback.right)
    bprev = Button(axprev, 'Left')
    bprev.on_clicked(callback.left)
    bexit = Button(axexit,'Save')
    bexit.on_clicked(callback.exit)
    bdump = Button(axdump,'Dump Population to Disk')
    bdump.on_clicked(callback.dump)

    plt.show()
    return callback.choice


def loadPopulation():
    global SHAPE, GENOME_LENGTH
    if path.exists("./population.p"):
        population = load(open("population.p","rb"))
    else:
        population = [Org() for _ in range(POP_SIZE)]
    return population


# PARAMETERS -------------------------------------------------------------------------------
# ---------- loaded in the global scope so they can be imported by other programs ----------
DATA_PATH = "./Data/thumbnails128x128-20210308T030619Z-001/thumbnails128x128"
POP_SIZE = 100
GENOME_LENGTH = 500
MUTATION_RATE = 0.01 #per site rate
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    U, S, Vh, SHAPE = loadData(DATA_PATH,GENOME_LENGTH)
    population = loadPopulation()

    mutants_kept = []
    generation = 0
    logdata = []
    aveFace = np.mean([org.genome for org in population], axis=0)

    while True:
        generation += 1
        # print("Generation:", generation)#,"\t MAX:", max(population, key= lambda org: org.get_fitness()))

        # (chooose x, mutate) cross (choose y, mutate) VS (choose z) (mix 2 compare with 3rd)
        newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),random.choices(population,k=1)[0].make_mutated_copy())
        selectionIndex = random.randint(0,POP_SIZE-1)
        testOrg = population[selectionIndex]
        testGroup = [newOrg,testOrg]
        shuffle = random.randint(0,1)
        winner = imageTournament(testGroup[shuffle],testGroup[1-shuffle]) #if shuffle == 0, mutant is left, if 1, right.

        #(chooose x, mutate) cross (choose y, mutate) VS (y)   (mix 2 compare with 2nd)
        # selectionIndex = random.randint(0,POP_SIZE-1)
        # Y = population[selectionIndex]
        # newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),Y.make_mutated_copy()) #"X" is the first parameter to recombine
        # testGroup = [newOrg,Y]
        # random.shuffle(testGroup)
        # winner = imageTournament(testGroup[0],testGroup[1])

        if winner == 2:
            print("Saving population to disk...")
            dump(population,open("population.p",'wb'))
            print("Save complete.")
        elif winner == 3:
            dumpPath = "./DUMP_{}".format(datetime.now().__str__().replace("-","_").replace(":","_").replace(".","_").replace(" ","_"))
            mkdir(dumpPath)
            for i,org in enumerate(population):
                plt.figure(figsize=(5,5))
                image = np.reshape(np.array(np.clip(np.dot(org.genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
                plt.imshow(image,interpolation="quadric")
                plt.savefig("{}/{}.png".format(dumpPath,i))
                plt.close()
        elif winner == -1:
            print("Window Closed. Exiting...")
            saveChoice = input("Save Population? [y/N]:")
            if saveChoice in ["y","yes"]:
                print("Saving population to disk...")
                dump(population,open("population.p",'wb'))
                print("Save complete.")
            else:
                print("Exited without saving.")
            exit()
        else:
            if winner == 0: #left choice
                population[selectionIndex] = testGroup[shuffle]
                mutants_kept.append(1-shuffle)
            else: #right choice == 1
                population[selectionIndex] = testGroup[1-shuffle]
                mutants_kept.append(shuffle)

        if len(mutants_kept) == 25:
            logdata.append(sum(mutants_kept)/25)
            mutants_kept.pop(0)

        aveFace = np.mean([org.genome for org in population], axis=0)