## IMPORTS

import os
import pickle
import numpy as np
import pynmrstar

from tqdm import tqdm

from os.path import exists

## FUNCTIONS

def collect_dataset(files:list):
    """
    Collect a dataset from a list of filepaths and save as a pickle archive
    """

    pdb_ids = []
    chemical_shift_tensors = []
    temperatures = []
    pHs = []

    pdb_id = None

    for file in tqdm(files):
        # Setup the entry
        entry = pynmrstar.Entry.from_file(
            file,
            convert_data_types=True
        )

        pdb_id = None

        # Get accession code
        tags = ["Accession_code"]

        pdb_id = [db_loop.get_tag(tags) for db_loop in entry.get_loops_by_category("Assembly_db_link")]
        
        #print(len(pdb_id))
        #assert len(pdb_id) == 1 or 0 # should be a single entry in there
        pdb_ids.append(pdb_id)

        # if the last accession code exists, (just added a PDB code):
        if len(pdb_id) == 1 :
            # Get the next chemical shift tensor
            tags = ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']        
            cs_result_sets = [chemical_shift_loop.get_tag(tags) for chemical_shift_loop in entry.get_loops_by_category("Atom_chem_shift")]
            
            chem_shifts = np.array(cs_result_sets)
            chemical_shift_tensors.append(chem_shifts)
            
    tensors = [t.squeeze() for t in chemical_shift_tensors]

    file = open("pdb_ids.pkl", 'wb')
    pickle.dump(pdb_ids, file)
    file.close()

    file = open("cs_tensors.pkl", 'wb')
    pickle.dump(chemical_shift_tensors, file)
    file.close() 

    return

def load_dataset(dataset_file:str="cs_tensors.pkl") -> list:
    """
    Loads a dataset from disk if not already stored. Else, loads by processing the data
    """

    if exists(dataset_file):
        file = open(dataset_file, 'rb')
        chemical_shifts = pickle.load(file)

        file = open("pdb_ids.pkl", 'rb')
        pdb_ids = pickle.load(file)
        
        print("SUCCESS:\t Datafile loaded from disk")
        return (chemical_shifts, pdb_ids)
    else:
        print("ERROR:\t Datafile not loaded. Please load manually")
        return None



### STRUCTURES FROM THE PDB
# pilfered from https://stackoverflow.com/questions/37335759/using-python-to-download-specific-pdb-files-from-protein-data-bank

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None

def download_dataset(pdb_ids:list, datadir:str):
    """
    Download a set of files corresponding to a list of input PDBs
    """
    for id in tqdm(pdb_ids):
        download_pdb(
            pdbcode=id, 
            datadir=datadir
        )