import copy
import random
from os import mkdir, path
from pathlib import Path
from pickle import dump, load
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
from statistics import mean

#important references
#https://sandipanweb.wordpress.com/2018/01/06/eigenfaces-and-a-simple-face-detector-with-pca-svd-in-python/
#https://matplotlib.org/stable/gallery/event_handling/keypress_demo.html#sphx-glr-gallery-event-handling-keypress-demo-py


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


def loadPopulation():
    global SHAPE, GENOME_LENGTH
    if path.exists("./population.p"):
        population = load(open("population.p","rb"))
    else:
        population = [Org() for _ in range(POP_SIZE)]
    return population


def save_truncated_svd(U,S,Vh,cutoff):
    dump((U[:,:cutoff], S[:cutoff], Vh[:cutoff,:]),open("./Preprocess/svd_truncated_{}.p".format(cutoff),'wb'))


def clamp(val):
    return min(1,max(0,val))

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


def prepareNextChoice():
    global mutants_kept, logdata,selectionIndex, shuffleBit, testGroup, plt1, plt2, plt3, plt4, axs, fig, population
    #reduce the keep log to only the most recent 25 events
    if len(mutants_kept) > 20:
        mutants_kept.pop(0)
    #append new average to average log
    logdata.append(mean(mutants_kept))
    #create a sliding window effect (fixes annoying bug (feature?) in matplotlib) https://github.com/matplotlib/matplotlib/issues/8516/
    logdata.pop(0)
    
        
    #update the average face image given the new population
    aveFace = np.mean([org.genome for org in population], axis=0)

    # (chooose x, mutate) cross (choose y, mutate) VS (choose z) (mix 2 compare with 3rd)
    #generate candidate organism
    newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),random.choices(population,k=1)[0].make_mutated_copy())
    #select competator
    selectionIndex = random.randint(0,POP_SIZE-1)
    testOrg = population[selectionIndex]
    #make the test blind to the user
    testGroup = [newOrg,testOrg]
    shuffleBit = random.randint(0,1)
    

    #(chooose x, mutate) cross (choose y, mutate) VS (y)   (mix 2 compare with 2nd)
    # #select competator
    # selectionIndex = random.randint(0,POP_SIZE-1)
    # Y = population[selectionIndex]
    # #generate candidate organism
    # newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),Y.make_mutated_copy())
    # #make the test blind to the user
    # testGroup = [newOrg,Y]
    # shuffle = random.randint(0,1)
    


    #refresh the display and await input
    img1 = np.reshape(np.array(np.clip(np.dot(testGroup[shuffleBit].genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE) #TODO: the need to manually truncate is obsolete
    img2 = np.reshape(np.array(np.clip(np.dot(testGroup[1-shuffleBit].genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
    imgAve = np.reshape(np.array(np.clip(np.dot(aveFace, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)

    plt1.set_ydata(logdata)
    axs[0][0].set_ylim( -0.05 , max(logdata)+0.05 ) #fixes bugged automatic limit changing in pyplot
    plt2.set_data(imgAve)
    plt3.set_data(img1)
    plt4.set_data(img2)

    fig.canvas.draw()


def on_press(event):
    global mutants_kept,selectionIndex, shuffleBit, population,testGroup

    if event.key == "left":
        population[selectionIndex] = testGroup[shuffleBit]
        mutants_kept.append(1-shuffleBit) #+1 if shuffle false
        prepareNextChoice()

    elif event.key == "right":
        population[selectionIndex] = testGroup[1-shuffleBit]
        mutants_kept.append(shuffleBit) #+1 if shuffle 1
        prepareNextChoice()

    elif event.key == "z":
        print("Saving population to disk...")
        sys.stdout.flush()
        dump(population,open("population.p",'wb'))
        print("Save complete.")
        sys.stdout.flush()

    elif event.key == "p":
        print("Dumping population images to disk...")
        sys.stdout.flush()
        dumpPath = "./DUMP_{}".format(datetime.now().__str__().replace("-","_").replace(":","_").replace(".","_").replace(" ","_"))
        mkdir(dumpPath)
        for i,org in enumerate(population):
            plt.figure(figsize=(5,5))
            image = np.reshape(np.array(np.clip(np.dot(org.genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
            plt.imshow(image,interpolation="quadric")
            plt.savefig("{}/{}.png".format(dumpPath,i))
            plt.close()
        print("Dump complete.")
        sys.stdout.flush()


# PARAMETERS -------------------------------------------------------------------------------
# ---------- loaded in the global scope so they can be imported by other programs ----------
DATA_PATH = "./Data/thumbnails128x128-20210308T030619Z-001/thumbnails128x128"
POP_SIZE = 100
GENOME_LENGTH = 500
MUTATION_RATE = 0.01 #per site rate
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #load data
    U, S, Vh, SHAPE = loadData(DATA_PATH,GENOME_LENGTH)
    population = loadPopulation()

    #init data logs
    mutants_kept = []
    logdata = [0 for _ in range(500)]
    aveFace = np.mean([org.genome for org in population], axis=0)


    # (chooose x, mutate) cross (choose y, mutate) VS (choose z) (mix 2 compare with 3rd)
    #generate first candidate organism
    newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),random.choices(population,k=1)[0].make_mutated_copy())
    #select first competator
    selectionIndex = random.randint(0,POP_SIZE-1)
    testOrg = population[selectionIndex]
    #make the test blind to the user
    testGroup = [newOrg,testOrg]
    shuffleBit = random.randint(0,1)
    

    #(chooose x, mutate) cross (choose y, mutate) VS (y)   (mix 2 compare with 2nd)
    # #select first competator
    # selectionIndex = random.randint(0,POP_SIZE-1)
    # Y = population[selectionIndex]
    # #generate first candidate organism
    # newOrg = recombine(random.choices(population,k=1)[0].make_mutated_copy(),Y.make_mutated_copy())
    # #make the test blind to the user
    # testGroup = [newOrg,Y]
    # shuffle = random.randint(0,1)
    


    #initalize the display window and await keyboard input
    img1 = np.reshape(np.array(np.clip(np.dot(testGroup[shuffleBit].genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE) #TODO: the need to manually truncate is obsolete
    img2 = np.reshape(np.array(np.clip(np.dot(testGroup[1-shuffleBit].genome, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
    imgAve = np.reshape(np.array(np.clip(np.dot(aveFace, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches((10, 10))

    fig.canvas.mpl_connect('key_press_event', on_press)

    axs[0][0].set_title("Rate of Change")
    plt1, = axs[0][0].plot(logdata)

    axs[0][1].set_title("Population Average")
    plt2 = axs[0][1].imshow(imgAve, interpolation="quadric")

    axs[1][0].set_title("Left Choice")
    plt3 = axs[1][0].imshow(img1, interpolation="quadric")

    axs[1][1].set_title("Right Choice")
    plt4 = axs[1][1].imshow(img2, interpolation="quadric")
    
    fig.suptitle("Press 'z' to save, 'p' to dump population, or left and right arrow keys to chose.")

    plt.show()

    #from here, evolution proceeds in between button press events, please see the on_press function for further details.

    print("Window Closed. Exiting...")
    saveChoice = input("Save Population? [y/N]:")
    if saveChoice in ["y","yes"]:
        print("Saving population to disk...")
        dump(population,open("population.p",'wb'))
        print("Save complete.")
    else:
        print("Exited without saving.")