import numpy as np
import sys

def read_oxdna(myfile):

    f = open(myfile, "r")

    trash = f.readline()  ##skip first line
    trash = f.readline()
    elems = trash.split(" ")
    box = np.asarray([float(elems[2]), float(elems[3]), float(elems[4].strip("\n"))])
    trash = f.readline()  ## skip third line

    data = []
    for line in f:
        elems = line.split(" ") 
        data.extend([float(e) for e in elems])

    f.close()
    w = len(elems)

    data = np.reshape(np.asarray(data), (len(data) // w, w))

    return box, data

def read_topology(myfile):

    f = open(myfile, "r")

    trash = f.readline()  ##skip first line
    data = []

    for line in f:
        elems = line.strip("\n").split(" ")
        data.extend([e for e in elems])
    
    f.close()
    w = len(elems)
    
    return np.reshape(np.asarray(data), (len(data) // w, w))

def print_oxdna(myfile, box, data):

    f = open(myfile, "w")

    f.write("t = 0\nb = %.3f %.3f %.3f\nE = 0 0 0\n" % (box[0], box[1], box[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write("%.12f " % data[i,j])
        f.write("\n")

    f.close()

def print_topo_oxdna(myfile, otop, Nf):

    seq = otop[:,1]
    mols = otop[:,0].astype(int)
    bonds = topol[:,-2:].astype(int)
    len_ss = seq.size // 2

    f = open(myfile, "w")

    f.write("%d %d\n" % (2*len_ss*Nf, 2*Nf))

    for i in range(Nf):
        for k in [0,1]:
            f.write("%d %s %d %d\n" % (mols[k*len_ss]+2*i, seq[k*len_ss], bonds[k*len_ss,0], bonds[k*len_ss,1]+2*len_ss*i))
            for j in range(1,len_ss-1):
                f.write("%d %s %d %d\n" % (2*i+k+1, seq[k*len_ss+j], bonds[k*len_ss+j,0]+2*len_ss*i, bonds[k*len_ss+j,1]+2*len_ss*i))
            f.write("%d %s %d %d\n" % (2*i+k+1, seq[k*len_ss+len_ss-1], bonds[k*len_ss+len_ss-1,0]+2*len_ss*i, bonds[k*len_ss+len_ss-1,1] ))


    f.close()


###########################################

if len(sys.argv) != 5 or sys.argv[1] == '-h':
    print("Usage is python3 ", sys.argv[0], 'file topology_file new_box_size N_filaments'); 
    exit(1)

myfile = sys.argv[1]
mytopof = sys.argv[2]
newbox = float(sys.argv[3])
N_filaments = int(sys.argv[4])

###########  data ##################

box, data = read_oxdna(myfile)

new_box = np.asarray([newbox, newbox, newbox])

com = np.average(data[:,:3], axis=0)

newdata = np.zeros((N_filaments*data.shape[0], data.shape[1]))

Ncells = int(np.ceil(N_filaments**0.33))
Lcell = new_box/(Ncells)

_square_lattice = [np.asarray([(ix+0.5)*Lcell[0], (iy+0.5)*Lcell[1], (iz+0.5)*Lcell[2]]) for iz in range(Ncells) for iy in range(Ncells) for ix in range(Ncells)]

for i in range(N_filaments):
    newdata[i*data.shape[0]:(i+1)*data.shape[0],:3] = data[:,:3] + _square_lattice[i] - com
    newdata[i*data.shape[0]:(i+1)*data.shape[0],3:] = data[:,3:]

print("Printing files restart.dat and topol.top in the current folder")
print_oxdna("restart.dat", new_box, newdata)
###############################################

topol = read_topology(mytopof)

print_topo_oxdna("topol.top", topol, N_filaments)




