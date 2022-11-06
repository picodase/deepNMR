## IMPORTS

import os, sys
import urllib
import pickle
import numpy as np
import pandas as pd
import pynmrstar

from tqdm import tqdm

from os.path import exists

## FUNCTIONS

def search_files(directory='.', extension=''):
    filelist = []

    extension = extension.lower()
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                filelist.append(os.path.join(dirpath, name))
            elif not extension:
                #print(os.path.join(dirpath, name))
                continue
    return filelist

def pickle_variable(variable, varname="variable.pkl"):
    """
    Pickles a variable into a given filename
    """

    file = open(varname, 'wb')
    pickle.dump(variable, file)
    file.close()
    return

def unpickle_variable(datafile="variable.pkl"):
    """
    Unpickles a variable into a given filename
    """
    file = open(datafile, 'rb')
    variable = pickle.load(file)
    file.close()

    return variable

def collect_dataset(files:list):
    """
    Collect a dataset from a list of filepaths and save as a pickle archive
    """

    pdb_ids = []
    chemical_shift_tensors = []
    conditions = {
        "pH": [],
        "temperature": []
    }

    element_sets = []

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
        if len(pdb_id) >= 1 :
            # Get the next chemical shift tensor
            tags = ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']        
            cs_result_sets = [chemical_shift_loop.get_tag(tags) for chemical_shift_loop in entry.get_loops_by_category("Atom_chem_shift")]
            
            chem_shifts = np.array(cs_result_sets)[0]

            #print(chem_shifts)

            #print(chem_shifts.shape)

            df = pd.DataFrame(
                data=chem_shifts, 
                columns=["res_idx", "res_id", "atom_type", "element", "chem_shift", "cs_error"]
            )

            elems = set(df["element"])

            chemical_shift_tensors.append(df)

            element_sets.append(elems)

            condn = []

            tags = ['Type', 'Val']
            for condition in entry.get_loops_by_category("Sample_condition_variable"):
                condn.append(condition.get_tag(tags))


            ### NOTE: Still need to fix:
            # - This code DOES NOT resolve by PDB ID, so it just makes a bunch of lists

            for c in condn[0]:
                label = c[0]

                if label in conditions: 
                    conditions[label].append(c[1])
                else: 
                    conditions[label] = []

    #tensors = [t.squeeze() for t in chemical_shift_tensors]

    pickle_variable(varname="pdb_ids.pkl", variable=pdb_ids)
    pickle_variable(varname="cs_tensors.pkl", variable=chemical_shift_tensors)
    pickle_variable(varname="elems.pkl", variable=element_sets)
    pickle_variable(varname="conditions.pkl", variable=conditions)

    """
    file = open("pdb_ids.pkl", 'wb')
    pickle.dump(pdb_ids, file)
    file.close()

    file = open("cs_tensors.pkl", 'wb')
    pickle.dump(chemical_shift_tensors, file)
    file.close() 

    file = open("elems.pkl", 'wb')
    pickle.dump(element_sets, file)
    file.close()
    """
    return


def load_dataset(dataset_file:str="cs_tensors.pkl") -> list:
    """
    Loads a dataset from disk if not already stored. Else, loads by processing the data
    """

    if exists(dataset_file):
        chemical_shifts = unpickle_variable(datafile=dataset_file)
        pdb_ids = unpickle_variable(datafile="pdb_ids.pkl")
        element_sets = unpickle_variable(datafile="elems.pkl")
        conditions = unpickle_variable(datafile="conditions.pkl")
        
        print("SUCCESS:\t Datafiles loaded from disk")
        return (chemical_shifts, pdb_ids, element_sets, conditions)
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