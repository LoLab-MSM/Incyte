# @sbeik

import os
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.optimize as opt
import scipy

def read_all_plates_and_split(path_to_files, split_path='hts_split_files'):
    '''
        Read in all the plates in all the files and break each plate into its own file by cell type, plate type,
        timepoint, and assay. These will be temporary files.

    :param path_to_files: Path to where the files are with all the plate readings
    :param split_path: Path to split the files to (this dir will be removed later)
    :return: split_path, to be used to read the files back in later
    '''

    if not os.path.exists(split_path):
        os.makedirs(split_path)

    for p in glob.glob(os.path.join(path_to_files, '*')):
        currgroup_l = []
        currgroup_c = []
        cell_type = None
        lum_or_ct = None
        timepoint = None
        startover = False
        with open(p) as f:
            for line in f:
                if "Field Group" in line:
                    if len(currgroup_l) > 0:      # if the first time 'field group' appears, no plate has been recorded yet
                        # if any time after the first, both plates have been read through, so start everything over
                        fn = os.path.join(split_path,(str(cell_type)+'-'+str(plate_type)+'-'+str(timepoint)+'-lum.txt'))
                        with open(fn,'w') as fout:
                            for i in currgroup_l:
                                fout.write(i)
                        startover = True
                    if len(currgroup_c) > 0:  # if the first time 'field group' appears, no plate has been recorded yet
                        fn = os.path.join(split_path,
                                          (str(cell_type) + '-' + str(plate_type) + '-' + str(timepoint) + '-Celltox.txt'))
                        with open(fn, 'w') as fout:
                            for i in currgroup_c:
                                fout.write(i)
                        startover = True
                    if startover:
                        currgroup_l = []
                        currgroup_c = []
                        cell_type = None
                        lum_or_ct = None
                        timepoint = None
                        startover = False
                elif 'Barcode:' in line:
                    cell_info = line.split('\t')[1]
                    cell_type = cell_info.split('-')[0]
                    plate_type = cell_info.split('-')[1]
                    timepoint = cell_info.split('-')[2]
                elif 'lum' in line.split('\t')[0] or 'Celltox' in line.split('\t')[0]: # must be a better way to do this
                    lum_or_ct = line.split('\t')[0].split(':')[0]
                else:
                    if lum_or_ct == 'lum':
                        currgroup_l.append(line)
                    elif lum_or_ct == 'Celltox green':
                        currgroup_c.append(line)

    return split_path


def read_drugs_table(plate, table_type, drug_list, conc_list):
    '''
        This takes a pandas dataframe of the exact plate and returns a dict-of-dicts with the information about
        each cell's drug's concentrations and replicates. Works for table types 1 through 4 (as discussed 7/7/16).

    :param plate: pandas dataframe denoting the plate with the cells treated by different drug concentrations
    :param table_type: plate type (1 through 4), to make the correct map of drugs and drug concentrations that the
                       replicates are treated with
    :param drug_list: list of drugs (in order) used on the plate - order specific for table type being passed in
    :param conc_list: list of concentrations used on the plate per 10 wells, in order left-to-right
                      concentrations should be floats, so need to have picked one standard unit to use
    :return: a dictionary of dictionaries with drug concentrations and their effects on replicates
    '''

    ### Map to connect how the drugs and drug concentrations are arranged on the plate with
    ##  the drugs and drug concentrations to be returned as a dictionary. Will be different depending on the
    ##  table_type passed in.

    drug_map = {}
    if table_type in ['2','3']:
        # - example:
        #   drug #1 (the first drug in the drug_list, indexed 0 here) is used in row B, wells 3-12, row K, wells 3-12,
        #   and row F, wells 13-22 (well indices here (given to range()) are off by 1 with actual well names on the
        #   plate)
        drug_map[0] = [('B', range(2, 12)), ('K', range(2, 12)), ('F', range(12, 22))]
        # and so on
        drug_map[1] = [('C', range(2, 12)), ('L', range(2, 12)), ('G', range(12, 22))]
        drug_map[2] = [('D', range(2, 12)), ('M', range(2, 12)), ('H', range(12, 22))]
        drug_map[3] = [('E', range(2, 12)), ('N', range(2, 12)), ('I', range(12, 22))]
        drug_map[4] = [('F', range(2, 12)), ('O', range(2, 12)), ('J', range(12, 22))]
        drug_map[5] = [('G', range(2, 12)), ('B', range(12, 22)), ('K', range(12, 22))]
        drug_map[6] = [('H', range(2, 12)), ('C', range(12, 22)), ('L', range(12, 22))]
        drug_map[7] = [('I', range(2, 12)), ('D', range(12, 22)), ('M', range(12, 22))]
        drug_map[8] = [('J', range(2, 12)), ('E', range(12, 22)), ('N', range(12, 22))]
    elif table_type in ['1','4']:
        # - example:
        #   drug #1 (the first drug in the drug_list, indexed 0 here) is used in rows B, C, and D, in  wells 3-12
        #   (well indices here (given to range()) are off by 1 with actual well names on the plate)
        drug_map[0] = [('B', range(2, 12)), ('C', range(2, 12)), ('D', range(2, 12))]
        # and so on
        drug_map[1] = [('E', range(2, 12)), ('F', range(2, 12)), ('G', range(2, 12))]
        drug_map[2] = [('H', range(2, 12)), ('I', range(2, 12)), ('J', range(2, 12))]
        drug_map[3] = [('K', range(2, 12)), ('L', range(2, 12)), ('M', range(2, 12))]
        drug_map[4] = [('N', range(2, 12)), ('O', range(2, 12)), ('B', range(12, 22))]
        drug_map[5] = [('C', range(12, 22)), ('D', range(12, 22)), ('E', range(12, 22))]
        drug_map[6] = [('F', range(12, 22)), ('G', range(12, 22)), ('H', range(12, 22))]
        drug_map[7] = [('I', range(12, 22)), ('J', range(12, 22)), ('K', range(12, 22))]
        drug_map[8] = [('L', range(12, 22)), ('M', range(12, 22)), ('N', range(12, 22))]

    ### Make the dictionary to be returned. For plate styles 1 through 4 there is a certain arrangement of media,
    ##  cells alone, and DMSO, so record it in the plate dictionary.
    ##
    ## Top two lines for each media/DMSO/cells alone is the method currently in use, separating controls based on the
    ## plate from which they came. In analysis, the user will have to account for which control goes with which drug
    ## (will need to know which plate the drug came from). The bottom two lines, commented out, are the options for
    ## pooling all controls of the same celltype, assay (celltox or luminescence), and timepoint to compare to whatever
    ## drug is also used on that cell type for that assay at that timepoint.

    plate_dict = {}
    # rows 1 and 24 are just media
    plate_dict[('Media-No_cells:'+str(table_type))] = {}
    plate_dict[('Media-No_cells:'+str(table_type))][0] = list(plate['1']) + list(plate['24'])
    #plate_dict[('Media-No_cells')] = {}
    #plate_dict['Media-No_cells'][0] = list(plate['1'])+list(plate['24'])

    # column 23 is cells treated with DMSO, as is the second part of row O before the last 2 columns
    plate_dict[('DMSO:'+str(table_type))] = {}
    plate_dict[('DMSO:'+str(table_type))]['N_A'] = list(plate['23'])+list(plate.loc['O'][12:22])
    #plate_dict['DMSO'] = {}
    #plate_dict['DMSO']['N_A'] = list(plate['23'])+list(plate.loc['O'][12:22])

    # rows A and P are cells alone, except for the first column and last 2 columns - ignore the 2nd column while
    # grabbing these row values to not overlap when grabbing column 2 (we don't want repeated wells)
    plate_dict[('None:'+str(table_type))] = {}
    plate_dict[('None:'+str(table_type))][0] = list(plate.loc['A'][2:22])+list(plate.loc['P'][2:22])+list(plate['2'])
    #plate_dict['None'] = {}
    #plate_dict['None'][0] = list(plate.loc['A'][2:22])+list(plate.loc['P'][2:22])+list(plate['2'])

    ### Now loop through the provided list of drugs (must be in the same order as the drugs on the plate) and the
    ##  provided list of drug concentrations (must be in the same order left-to-right as drug concentrations on the
    ##  plate) and use drug_map to find each drug's half-rows and each drug concentration's wells within those half-rows

    for i in range(0,len(drug_list)):
        plate_dict[drug_list[i]] = {}
        for j in range(0,len(conc_list)):
            plate_dict[drug_list[i]][conc_list[j]] = []
            # drug map indices correspond with drug list indices (ie drug_map[0] is for first given drug, drug_list[0]
            # is for first given drug)
            for map_row in drug_map[i]:
                # plate row name is given by map_row[0] (ex. 'B')
                # which half of the plate (wells 3-12 or wells 13-22) given by map_row[1]
                # map_row[1], which gives the range of wells, drug concs correspond to the list of concentrations given
                #  by conc_lists (conc_lists order must match up with order of drug concentrations on the plate)
                plate_dict[drug_list[i]][conc_list[j]].append(plate.loc[map_row[0]][map_row[1]][j])

    return plate_dict


def build_dict_of_dicts(path_to_files, drug_dict, conc_list):
    '''

        This reads in all of the temporarily made files (of the different plates) as panda arrays, then calls other
        functions to turn the plate-file matrices into a dictionary of dictionaries of drugs and drug concentrations
        per a cell type's replicates. These nested dictionaries are returned here, and then this function builds the
        rest of the nested dictionary with the cell type, assay type, and timepoint information (gained from the temporary
        files' names).

        Right now can handle plate types 2, 3, 4 (calls the methods that will map to these plates)

    :param path_to_files: path to the directory in which the temporary files can be found
    :return: nested dictionary d, of the format
             {cell type:{assay type:{timepoint:{drug:{drug concentration:(replicates)}}}}}
    '''

    d = {}
    for f in glob.glob(os.path.join(path_to_files, '*')):
        ## Gain the information you can from the filename (celltype, plate type, timepoint, assay) and make
        #  dict keys as needed
        cell_info = os.path.basename(f).split('.txt')[0]
        cell_type = cell_info.split('-')[0]
        if not cell_type in d:
            d[cell_type] = {}
        print(cell_info)
        assay = cell_info.split('-')[3]
        if not assay in d[cell_type]:
            d[cell_type][assay] = {}
        # Make timepoint a number (for this will need to standardize the units, in this case always use hours)
        timepoint = float(cell_info.split('-')[2].split('hr')[0])
        if not timepoint in d[cell_type][assay]:
            d[cell_type][assay][timepoint] = {}
        plate_type = cell_info.split('-')[1]

        ## Read in the table appropriately to pandas
        # the final column doesn't have data, it either says 'lum:Lum' or 'Celltox green:485,528'
        # --> usecols eliminates the final column
        # once you eliminate the final column, matrix will have one more column of data than column names
        # (since first column is row names) - so index_col is set to 0 to show the first col is the row names
        x = pd.read_table(f, sep='\t', usecols=range(0, 25), index_col=0)

        ## Send the table to the function that will map the drugs and drug concentrations per replicate of this cell
        #  type for this assay at this timepoint. Need to have the right plate_type number for the list of drugs.
        xdict = read_drugs_table(x,plate_type,drug_dict[plate_type],conc_list)

        ## Put the dict of drugs, drug concentrations, and values of the replicates treated with those concentrations
        #  into the dict of dicts to return for the timepoint, assay type, and cell type.

        for i in xdict:
            if not i in d[cell_type][assay][timepoint]:
                d[cell_type][assay][timepoint][i] = xdict[i]
            else:
                print cell_type + ' ' + assay + ' ' + str(timepoint) + ' ' + str(plate_type) + ' ' + i
                # If that drug name is already in the dictionary, it means that:
                # The same drug was used 1) on this cell type 2) for this assay 3) at this timepoint 4) on a diff plate,
                # giving different (well, additional) results. Could use the same OR different concentrations.
                # 1) check if the concentrations are the same
                #   1a) if different, just add the new concentration's results to the dict
                #   1b) if the same concentration, add the results to the list ([].extend())
                for j in xdict[i]: # for each concentration in the returned xdict for the drug we're checking
                    if not j in d[cell_type][assay][timepoint][i]:
                        # if the concentration isn't in the dict already, put it in
                        d[cell_type][assay][timepoint][i][j] = xdict[i][j]
                    else:
                        # if the concentration is already present, include its cell measurements
                        d[cell_type][assay][timepoint][i][j].extend(xdict[i][j])

    return d


def old_same_ct_diff_drug(ct_dict, cell_type, conc_list, drug_list, timepoint, assay, control=('DMSO:2','N_A')):
    '''

    :param ct_dict: Nested dictionary for the desired cell type (as in, this cell type's subset of the overall
    nested dictionary).
    :param cell_type: The cell type of interest, for plot title purposes - should be a string.
    :param conc_list: the list of concentrations (drug dilutions) to be comparedthe cells are treated with across the
    desired cell types at the desired timepoint for the desired assay - should be a list of floats
    :param drug_list: a list of the drugs to compare across differing dilutions for the desired cell type, at the
    correct timepoint; only drugs known to be tested on the desired cell type should be included - should be a
    list of strings
    :param timepoint: the timepoint of interest - should be a float
    :param assay: either 'lum' for the luminesence assay of cell viability, or 'Celltox' for the Celltox green assay of
    cell toxicity - is a string
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)
    '''

    ## To make the dataframe, will need a dictionary with the drugs as the keys and as their values the flattened,
    #  ordered (by smallest->largest dilution) measurements (either viability or cell toxicity) for that drug as one
    #  list. The lists that will become the dictionary values must all be the same length, so first interrogate the
    #  number of replicates per drug per drug dilution. The largest number of replicates found will be used as the
    #  number 'expected' for all cell types for all drug dilutions, to ensure list length equality.

    num_reps = []
    for i in drug_list:
        for j in ct_dict[assay][timepoint][i]:
            num_reps.append(len(ct_dict[assay][timepoint][i][j]))  # make list of all replicate numbers
    num_rep = max(num_reps)

    # 'rep_list' is the list of replicates that will be used to index the data frame. Each index in rep_list corresponds
    # to one replicate plate-well measurement value. (If a cell type/drug treatment/drug dilution does not have enough
    # replicates, it will be handled.)
    rep_list = list(np.arange(num_rep))

    # 'concs_sorted_expanded' is the list of drug dilutions that will be used to index the dataframe along with
    # 'rep_list'. However, there are (usually) multiple replicates per drug dilution. Thus, each drug dilution will be
    # repeated for as many replicates as are tested at this dilution. The list is sorted to make sure the dilution in
    # the dataframe index matches the dilution used in the plate-well from which the measurement was taken.
    concs_sorted_expanded = sorted(conc_list * num_rep)

    ## Now make the dictionary with the drugs of interest as the keys and as the values the flattened, ordered list of
    #  measurements of cell type+drug at each dilution. Sort the keys so that the correct dilution's measurements are
    #  accessed from the overall nested dictionary, leading to correctly ordered measurements in the dataframe per drug
    #  type that match up with dilutions in the dataframe index.

    valuedict_from_sorted_keys = {}
    for i in drug_list:
        # values will come from the overall nested dictionary, so each drug is only represented once
        valuedict_from_sorted_keys[i] = []
        for conc in sorted(list(set(concs_sorted_expanded))):
            # make sure the dilution in the list has been measured for this cell type/assay/timepoint/drug
            if conc in ct_dict[assay][timepoint][i]:
                valuedict_from_sorted_keys[i].extend(ct_dict[assay][timepoint][i][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as other drugs
                # passed in, supplement the measurements it does have with 'None' values
                if not len(ct_dict[assay][timepoint][i][conc]) == num_rep:
                    valuedict_from_sorted_keys[i].extend(
                        [None] * (num_rep - len(ct_dict[assay][timepoint][i][conc])))
            # if the dilution in the list has not been measured for this cell type/assay/time/drug
            else:
                # give it as many 'None's as necessary to match up with other measured drugs
                # note - a cell type with all Nones will look weird in the resulting plot
                valuedict_from_sorted_keys[i].extend([None] * num_rep)

    ## Now make the dataframe, indexed both by dilutions (given in conc_list) and by replicates (determined during this
    #  function). For the concentration list given on 7/7/16, the index looks best as an x-axis when using the log (base
    #  10) of the concentrations - that way they're evenly spaced. Because of a bug in pandas/matplotlib, yerr can only
    #  be plotted when the index includes zero as one of its values, so I take the largest (in this case, least
    #  negative) value of the log10(concs) and subtract it from all log-concentration values, (in this case, subtracting
    #  a negative, thus adding that value). This means that log-concentration value is now zero, and all
    #  log-concentration values are shifted the same amount. (The correct log-concentration values will overwrite these
    #  incorrect values on the x-axis when the graph is plotted.)

    z = [math.log10(k * 10e-9) for k in concs_sorted_expanded]  # drug dilutions per replicate, log (base 10) value
    # make the multiple index for the dataframe, giving the list of drug dilutions per replicate, and replicate list
    # multiplied by the number of concentrations - this way both lists are the same length and can be used properly to
    # index the dataframe
    ix = pd.MultiIndex.from_arrays([list(np.subtract(z, [max(z)] * len(z))), rep_list * len(conc_list)],
                                   names=['dilutions', 'replicates'])
    # make the dataframe using the index we just made, and give as the dictionary input required the dictionary of each
    # drug and its flattened measurement values per replicate per dilution
    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## For this particular method we divide the plate-well measurement by the average of this measurement of the control
    #  cells for this cell type. The 'control' tuple gives the type of control ('None:x' for cells only, 'DMSO' for
    #  cells treated with DMSO, or 'Media-No_cells' for wells of only media) and the single dictionary key used by that
    #  type of control in the overall nested dict (cells only key is 0, DMSO key is 'N_A', media only key is 0 - see
    #  function 'read_drugs_table()')

    for i in df:
        df[i] = df[i] / np.mean(ct_dict[assay][timepoint][control[0]][control[1]])

    ## Group the drugs by dilutions, because we're going to take the mean of the replicates and leave the dilutions as
    #  they are (to make sure we get the mean for each one). Then calculate the mean and standard deviation, to use for
    #  the plot.

    gp = df.groupby(level=('dilutions'))
    means = gp.mean()
    errors = gp.std()

    ## Make the list of strings to use for the x-axis - the nontransformed values with which we'll replace the
    #  transformed values (again, transformed values are a hack to get around the pandas/matplotlib problem with yerr).
    #  This problem should be fixed (according to pandas github) with a July 2016 pandas update, 1.18.2 - not released
    #  as of this writing.

    zz = [str(round(i, 2)) for i in sorted(list(set(z)))]  # list of strings for correct x-axis labeling
    fig, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax, xticks=means.index)
    ax.get_xaxis().set_ticklabels(zz, rotation='vertical')
    ax.set_xlabel('log([Cmpd]), M')
    if assay == 'lum':
        ax.set_ylabel('Viability (% of control)')
    elif assay == 'Celltox':
        ax.set_ylabel('Cell toxicity (% of control)')
    ax.set_title(cell_type + ' ' + str(timepoint) + ' hr')
    home = os.path.expanduser("~")
    fig1 = os.path.join(home, ('Different_drugs_vs_control_' + cell_type + '_' + str(timepoint) + '_' + assay + '.png'))
    plt.savefig(fig1)
    plt.close()

def old_same_ct_drug_diff_time(ct_dict, cell_type, conc_list, drug, time_list, assay, control=('DMSO:2','N_A')):
    '''

    :param ct_dict: Nested dictionary for the desired cell type (as in, this cell type's subset of the overall
    nested dictionary).
    :param cell_type: The cell type of interest, for plot title purposes - should be a string.
    :param conc_list: the list of concentrations (drug dilutions) to be comparedthe cells are treated with across the
    desired cell types at the desired timepoint for the desired assay - should be a list of floats
    :param drug: the drug whose dilutions are being compared across timepoints for a certain cell type - should be a
    string
    :param time_list: a list of the timepoints over which to compare drug dilutions of the desired drug for the
    desired cell type; only times at which measurements have been taken for the desired cell type and the desired drug
    should be included - should be a list of floats
    :param assay: either 'lum' for the luminesence assay of cell viability, or 'Celltox' for the Celltox green assay of
    cell toxicity - is a string
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)
    :return:
    '''

    ## To make the dataframe, will need a dictionary with the timepoints as the keys and as their values the flattened,
    #  ordered (by smallest->largest dilution) measurements (either viability or cell toxicity) for that timepoint as
    #  one list. The lists that will become the dictionary values must all be the same length, so first interrogate the
    #  number of replicates per timepoint per drug dilution. The largest number of replicates found will be used as
    #  the number 'expected' for all cell types for all drug dilutions, to ensure list length equality.

    num_reps = []
    for i in time_list:
        for j in ct_dict[assay][i][drug]:
            num_reps.append(len(ct_dict[assay][i][drug][j]))  # make list of all replicate numbers
    num_rep = max(num_reps)

    # 'rep_list' is the list of replicates that will be used to index the data frame. Each index in rep_list corresponds
    # to one replicate plate-well measurement value. (If a cell type/drug treatment/drug dilution does not have enough
    # replicates, it will be handled.)
    rep_list = list(np.arange(num_rep))

    # 'concs_sorted_expanded' is the list of drug dilutions that will be used to index the dataframe along with
    # 'rep_list'. However, there are (usually) multiple replicates per drug dilution. Thus, each drug dilution will be
    # repeated for as many replicates as are tested at this dilution. The list is sorted to make sure the dilution in
    # the dataframe index matches the dilution used in the plate-well from which the measurement was taken.
    concs_sorted_expanded = sorted(conc_list * num_rep)

    ## Now make the dictionary with the times of interest as the keys and as the values the flattened, ordered list of
    #  measurements of cell type+drug at each dilution. Sort the keys so that the correct dilution's measurements are
    #  accessed from the overall nested dictionary, leading to correctly ordered measurements in the dataframe per time-
    #  point that match up with dilutions in the dataframe index.

    valuedict_from_sorted_keys = {}
    for i in time_list:
        # values will come from the overall nested dictionary, so each timepoint is only represented once
        valuedict_from_sorted_keys[str(i)+' hr'] = []
        for conc in sorted(list(set(concs_sorted_expanded))):
            # make sure the dilution in the list has been measured for this cell type/assay/timepoint/drug
            if conc in ct_dict[assay][i][drug]:
                valuedict_from_sorted_keys[str(i)+' hr'].extend(ct_dict[assay][i][drug][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as other timepoint(s)
                # passed in, supplement the measurements it does have with 'None' values
                if not len(ct_dict[assay][i][drug][conc]) == num_rep:
                    valuedict_from_sorted_keys[str(i)+' hr'].extend(
                        [None] * (num_rep - len(ct_dict[assay][i][drug][conc])))
            # if the dilution in the list has not been measured for this cell type/assay/time/drug
            else:
                # give it as many 'None's as necessary to match up with other measured timepoints
                # note - a cell type with all Nones will look weird in the resulting plot
                valuedict_from_sorted_keys[str(i)+' hr'].extend([None] * num_rep)

    ## Now make the dataframe, indexed both by dilutions (given in conc_list) and by replicates (determined during this
    #  function). For the concentration list given on 7/7/16, the index looks best as an x-axis when using the log (base
    #  10) of the concentrations - that way they're evenly spaced. Because of a bug in pandas/matplotlib, yerr can only
    #  be plotted when the index includes zero as one of its values, so I take the largest (in this case, least
    #  negative) value of the log10(concs) and subtract it from all log-concentration values, (in this case, subtracting
    #  a negative, thus adding that value). This means that log-concentration value is now zero, and all
    #  log-concentration values are shifted the same amount. (The correct log-concentration values will overwrite these
    #  incorrect values on the x-axis when the graph is plotted.)

    z = [math.log10(k * 10e-9) for k in concs_sorted_expanded]  # drug dilutions per replicate, log (base 10) value
    # make the multiple index for the dataframe, giving the list of drug dilutions per replicate, and replicate list
    # multiplied by the number of concentrations - this way both lists are the same length and can be used properly to
    # index the dataframe
    ix = pd.MultiIndex.from_arrays([list(np.subtract(z, [max(z)] * len(z))), rep_list * len(conc_list)],
                                   names=['dilutions', 'replicates'])
    # make the dataframe using the index we just made, and give as the dictionary input required the dictionary of each
    # timepoint and its flattened measurement values per replicate per dilution
    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## For this particular method we divide the plate-well measurement by the average of this measurement of the control
    #  cells for this cell type. The 'control' tuple gives the type of control ('None:x' for cells only, 'DMSO' for
    #  cells treated with DMSO, or 'Media-No_cells' for wells of only media) and the single dictionary key used by that
    #  type of control in the overall nested dict (cells only key is 0, DMSO key is 'N_A', media only key is 0 - see
    #  function 'read_drugs_table()')

    for i in df:
        df[i] = df[i] / np.mean(ct_dict[assay][float(i.split(' hr')[0])][control[0]][control[1]])

    ## Group the timepoints by dilutions, because we're going to take the mean of the replicates and leave the dilutions
    #  as they are (to make sure we get the mean for each one). Then calculate the mean and standard deviation, to use
    #  for the plot.

    gp = df.groupby(level=('dilutions'))
    means = gp.mean()
    errors = gp.std()

    ## Make the list of strings to use for the x-axis - the nontransformed values with which we'll replace the
    #  transformed values (again, transformed values are a hack to get around the pandas/matplotlib problem with yerr).
    #  This problem should be fixed (according to pandas github) with a July 2016 pandas update, 1.18.2 - not released
    #  as of this writing.

    zz = [str(round(i, 2)) for i in sorted(list(set(z)))]  # list of strings for correct x-axis labeling
    fig, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax, xticks=means.index)
    ax.get_xaxis().set_ticklabels(zz, rotation='vertical')
    ax.set_xlabel('log([Cmpd]), M')
    if assay == 'lum':
        ax.set_ylabel('Viability (% of control)')
    elif assay == 'Celltox':
        ax.set_ylabel('Cell toxicity (% of control)')
    ax.set_title(cell_type + ' ' + drug + ' timecourse')
    home = os.path.expanduser("~")
    fig2 = os.path.join(home, ('Timecourse_vs_control_' + cell_type + '_' + drug + '_' + assay + '.png'))
    plt.savefig(fig2)
    plt.close()


def new_cells_over_time(ct_dict, cell_type, conc_list, drug, time_list, assay, control=('DMSO:2','N_A')):
    '''

    :param ct_dict: Nested dictionary for the desired cell type (as in, this cell type's subset of the overall
    nested dictionary).
    :param cell_type: The cell type of interest, for plot title purposes - should be a string.
    :param conc_list: the list of concentrations (drug dilutions) the cells are treated with for the desired assay,
    to compare over time - should be a list of floats
    :param drug: the drug whose dilutions are being compared over time for a certain cell type and drug - should be a
    string
    :param time_list: a list of the timepoints over which to compare drug dilutions of the desired drug for the
    desired cell type; only times at which measurements have been taken for the desired cell type and the desired drug
    should be included - should be a list of floats
    :param assay: either 'lum' for the luminesence assay of cell viability, or 'Celltox' for the Celltox green assay of
    cell toxicity - is a string
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)
    '''


    ## To make the dataframe, will need a dictionary with the drug dilutions as the keys and as their values the
    #  flattened, ordered (by smallest->largest time) measurements (either viability or cell toxicity) for that
    #  concentration as one list. The lists that will become the dictionary values must all be the same length, so first
    #  interrogate the number of replicates per timepoint per drug dilution. The largest number of replicates found will
    #  be used as the number 'expected' for all cell types for all drug dilutions, to ensure list length equality.

    num_reps = []
    for i in time_list:
        for j in ct_dict[assay][i][drug]:
            num_reps.append(len(ct_dict[assay][i][drug][j]))  # make list of all replicate numbers

    # do this for control as well - this will make num_reps large, but it will sort itself out
    for i in time_list:
        num_reps.append(len(ct_dict[assay][i][control[0]][control[1]]))

    num_rep = max(num_reps)

    # 'rep_list' is the list of replicates that will be used to index the data frame. Each index in rep_list corresponds
    # to one replicate plate-well measurement value. (If a cell type/drug treatment/drug dilution does not have enough
    # replicates, it will be handled.)
    rep_list = list(np.arange(num_rep))

    # 'times_sorted_expanded' is the list of times that will be used to index the dataframe along with 'rep_list'.
    # There are (usually) multiple replicates per timepoint. Thus, each timepoint will be repeated for as many
    # replicates as are tested at this time. The list is sorted to make sure the time in the dataframe index matches the
    # time at which the measurement was taken.
    times_sorted_expanded = sorted([t/24 for t in time_list] * num_rep)

    ## Now make the dictionary with the times of interest as the keys and as the values the flattened, ordered list of
    #  measurements of cell type+drug at each dilution. Sort the keys so that the correct dilution's measurements are
    #  accessed from the overall nested dictionary, leading to correctly ordered measurements in the dataframe per time-
    #  point that match up with dilutions in the dataframe index.

    valuedict_from_sorted_keys = {}
    for conc in conc_list:
        # values will come from the overall nested dictionary, so each concentration is only represented once
        valuedict_from_sorted_keys[str(conc)] = []
        for t in sorted(list(set(times_sorted_expanded))):
            # make sure the concentration in the list of keys has been measured for this cell type/assay/timepoint/drug
            if conc in ct_dict[assay][t*24][drug]:
                valuedict_from_sorted_keys[str(conc)].extend(ct_dict[assay][t*24][drug][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as other timepoint(s)
                # passed in, supplement the measurements it does have with 'None' values
                if not len(ct_dict[assay][t*24][drug][conc]) == num_rep:
                    valuedict_from_sorted_keys[str(conc)].extend(
                        [None] * (num_rep - len(ct_dict[assay][t*24][drug][conc])))
            # if the dilution in the list has not been measured for this cell type/assay/time/drug
            else:
                # give it as many 'None's as necessary to match up with other measured timepoints
                # note - a cell type with all Nones will look weird in the resulting plot
                valuedict_from_sorted_keys[str(conc)].extend([None] * num_rep)

    ## Add control values to the df, to compare the cell counts to
    valuedict_from_sorted_keys[control[0].split(':')[0]] = []
    for t in sorted(list(set(times_sorted_expanded))):
        valuedict_from_sorted_keys[control[0].split(':')[0]].extend(ct_dict[assay][t*24][control[0]][control[1]])

    ## Now make the dataframe, indexed both by time measurements were taken and by replicates (determined during this
    #  function).

    ix = pd.MultiIndex.from_arrays([times_sorted_expanded, rep_list * len(time_list)],
                                   names=['timepoints', 'replicates'])


    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## Ideally we'll normalize the data by the cell count (/toxicity) at time zero. We don't have a time zero
    #  measurement, but we can calculate one using cell_count_at_time_zero(). We can assume this number is the same
    #  for the controls and for any drug-treated cells (at any concentration).

    cellcount_tzero = cell_count_at_time_zero(ct_dict, time_list, control)
    zeroes_dict = {}
    for conc in conc_list:
        zeroes_dict[str(conc)] = cellcount_tzero

    zeroes_dict[control[0].split(':')[0]] = cellcount_tzero

    nix = pd.MultiIndex.from_arrays([[0], [0]], names=['timepoints', 'replicates'])
    df = pd.DataFrame(zeroes_dict, index=nix).append(df)

    ## Normalize the data by time zero

    for i in df:  # we know that df and df_plot have the same columns, because we haven't changed those
        # minus one because these should start at "zero"
        df[i] = (df[i] / df.iloc[0][i])-1 # iloc[0] is the first row of the data frame - time zero

    ## Now we want to plot the data appropriately (experimental values only)

    gp = df.groupby(level=('timepoints'))
    means = gp.mean()
    errors = gp.std()

    for i in errors:
        errors.rename(columns={i: i + '-err'}, inplace=True)
    m_e = pd.concat([means, errors], axis=1, join_axes=[means.index])

    conc_list.append(control[0].split(':')[0])
    for i, col in zip(conc_list, sns.color_palette("husl", len(conc_list))):
        # will give the same fit as before because passing in the same x-vals (we're not including the "conc zero" val)
        st, sc, sk = scipy.interpolate.splrep(m_e.index.get_level_values(0).values, m_e[str(i)])
        plt.errorbar(m_e.index.get_level_values(0).values, m_e[str(i)], yerr=m_e[str(i) + '-err'], fmt='o', color=col,
                     label=str(i))
        plt.plot(m_e.index.get_level_values(0).values,
                 scipy.interpolate.splev(m_e.index.get_level_values(0).values, (st, sc, sk)), color=col, label=None)

    # Get graph axes, shrink the plot's width a little, then put a legend outside the plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([round(min(m_e.index.get_level_values(0).values) - 1),
              round(max(m_e.index.get_level_values(0).values) + 1)])
    plt.xlabel('Time (days)')
    if assay == 'lum':
        plt.ylabel('Cell growth (normalized to cell # at time zero)')
    elif assay == 'Celltox':
        plt.ylabel('Cell death (normalized to cell # at time zero')
    plt.title(cell_type + ' ' + drug + ' over time')

    home = os.path.expanduser("~")
    fig3 = os.path.join(home, (cell_type + '_' + drug + '_' + '_' + assay + '_timeseries.png'))
    plt.savefig(fig3)
    plt.close()


def old_viab_vs_celltox(ct_dict, cell_type, conc_list, drug, timepoint, control=('DMSO:2','N_A')):
    '''

    :param ct_dict: Nested dictionary for the desired cell type (as in, this cell type's subset of the overall
    nested dictionary).
    :param cell_type: The cell type of interest, for plot title purposes - should be a string.
    :param conc_list: the list of concentrations (drug dilutions) to be comparedthe cells are treated with across the
    desired cell types at the desired timepoint for the desired assay - should be a list of floats
    :param drug: the drug whose dilutions are being compared for the two different assays, for a certain cell type
    - should be a string
    :param timepoint: the timepoint of interest - should be a float
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)
    :return:
    '''

    ## To make the dataframe, will need a dictionary with the two assays as the keys and as their values the flattened,
    #  ordered (by smallest->largest dilution) measurements (either viability or cell toxicity) for those assays for the
    #  desired cell type and desired drug. The lists that will become the dictionary values must all be the same length,
    #  so first interrogate the number of replicates per timepoint per drug dilution. The largest number of replicates
    #  found will be used as the number 'expected' for all cell types for all drug dilutions, to ensure list length
    #  equality.

    num_reps = []
    for i in ['lum','Celltox']:
        for j in ct_dict[i][timepoint][drug]:
            num_reps.append(len(ct_dict[i][timepoint][drug][j]))  # make list of all replicate numbers
    num_rep = max(num_reps)

    # 'rep_list' is the list of replicates that will be used to index the data frame. Each index in rep_list corresponds
    # to one replicate plate-well measurement value. (If a cell type/drug treatment/drug dilution does not have enough
    # replicates, it will be handled.)
    rep_list = list(np.arange(num_rep))

    # 'concs_sorted_expanded' is the list of drug dilutions that will be used to index the dataframe along with
    # 'rep_list'. However, there are (usually) multiple replicates per drug dilution. Thus, each drug dilution will be
    # repeated for as many replicates as are tested at this dilution. The list is sorted to make sure the dilution in
    # the dataframe index matches the dilution used in the plate-well from which the measurement was taken.
    concs_sorted_expanded = sorted(conc_list * num_rep)

    ## Now make the dictionary with the two assays as the keys and as the values the flattened, ordered list of
    #  measurements of cell type+drug at each dilution. Sort the keys so that the correct dilution's measurements are
    #  accessed from the overall nested dictionary, leading to correctly ordered measurements in the dataframe per assay
    # that match up with dilutions in the dataframe index.

    valuedict_from_sorted_keys = {}
    for i in ['lum','Celltox']:
        # values will come from the overall nested dictionary, so each assay is only represented once
        if i == 'lum':
            valuedict_from_sorted_keys['Viability'] = []
        else:
            valuedict_from_sorted_keys[i] = []
        for conc in sorted(list(set(concs_sorted_expanded))):
            # make sure the dilution in the list has been measured for this cell type/assay/timepoint/drug
            if conc in ct_dict[i][timepoint][drug]:
                if i == 'lum':
                    valuedict_from_sorted_keys['Viability'].extend(ct_dict[i][timepoint][drug][conc])
                else:
                    valuedict_from_sorted_keys[i].extend(ct_dict[i][timepoint][drug][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as the other assay
                # passed in, supplement the measurements it does have with 'None' values
                if not len(ct_dict[i][timepoint][drug][conc]) == num_rep:
                    if i == 'lum':
                        valuedict_from_sorted_keys['Viability'].extend(
                            [None] * (num_rep - len(ct_dict[i][timepoint][drug][conc])))
                    else:
                        valuedict_from_sorted_keys[i].extend(
                            [None] * (num_rep - len(ct_dict[i][timepoint][drug][conc])))
            # if the dilution in the list has not been measured for this cell type/assay/time/drug
            else:
                # give it as many 'None's as necessary to match up with other measured timepoints
                # note - a cell type with all Nones will look weird in the resulting plot
                if i == 'lum':
                    valuedict_from_sorted_keys['Viability'].extend([None] * num_rep)
                else:
                    valuedict_from_sorted_keys[i].extend([None] * num_rep)

    ## Now make the dataframe, indexed both by dilutions (given in conc_list) and by replicates (determined during this
    #  function). For the concentration list given on 7/7/16, the index looks best as an x-axis when using the log (base
    #  10) of the concentrations - that way they're evenly spaced. Because of a bug in pandas/matplotlib, yerr can only
    #  be plotted when the index includes zero as one of its values, so I take the largest (in this case, least
    #  negative) value of the log10(concs) and subtract it from all log-concentration values, (in this case, subtracting
    #  a negative, thus adding that value). This means that log-concentration value is now zero, and all
    #  log-concentration values are shifted the same amount. (The correct log-concentration values will overwrite these
    #  incorrect values on the x-axis when the graph is plotted.)

    z = [math.log10(k * 10e-9) for k in concs_sorted_expanded]  # drug dilutions per replicate, log (base 10) value
    # make the multiple index for the dataframe, giving the list of drug dilutions per replicate, and replicate list
    # multiplied by the number of concentrations - this way both lists are the same length and can be used properly to
    # index the dataframe
    ix = pd.MultiIndex.from_arrays([list(np.subtract(z, [max(z)] * len(z))), rep_list * len(conc_list)],
                                   names=['dilutions', 'replicates'])
    # make the dataframe using the index we just made, and give as the dictionary input required the dictionary of each
    # timepoint and its flattened measurement values per replicate per dilution
    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## For this particular method we divide the plate-well measurement by the average of this measurement of the control
    #  cells for this cell type. The 'control' tuple gives the type of control ('None:x' for cells only, 'DMSO' for
    #  cells treated with DMSO, or 'Media-No_cells' for wells of only media) and the single dictionary key used by that
    #  type of control in the overall nested dict (cells only key is 0, DMSO key is 'N_A', media only key is 0 - see
    #  function 'read_drugs_table()')

    for i in df:
        if i == 'Viability':
            df[i] = df[i] / np.mean(ct_dict['lum'][timepoint][control[0]][control[1]])
        else:
            df[i] = df[i] / np.mean(ct_dict[i][timepoint][control[0]][control[1]])

    ## Group the assays by dilutions, because we're going to take the mean of the replicates and leave the dilutions
    #  as they are (to make sure we get the mean for each one). Then calculate the mean and standard deviation, to use
    #  for the plot.

    gp = df.groupby(level=('dilutions'))
    means = gp.mean()
    errors = gp.std()

    ## Make the list of strings to use for the x-axis - the nontransformed values with which we'll replace the
    #  transformed values (again, transformed values are a hack to get around the pandas/matplotlib problem with yerr).
    #  This problem should be fixed (according to pandas github) with a July 2016 pandas update, 1.18.2 - not released
    #  as of this writing.

    zz = [str(round(i, 2)) for i in sorted(list(set(z)))]  # list of strings for correct x-axis labeling
    fig, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax, xticks=means.index)
    ax.get_xaxis().set_ticklabels(zz, rotation='vertical')
    ax.set_xlabel('log([Cmpd]), M')
    ax.set_ylabel('Amount relative to control')
    ax.set_title(drug + ' (' + cell_type + ') ' + str(timepoint) + ' hr')
    home = os.path.expanduser("~")
    fig4 = os.path.join(home, ('Viability_vs_Celltox_' + cell_type + '_' + drug + '_' + str(timepoint) + '.png'))
    plt.savefig(fig4)
    plt.close()

def old_diff_ct_same_drug(total_dict, conc_list, drug, timepoint, cell_types, assay, control=('DMSO:2', 'N_A')):
    '''

        This function makes the type of plot discussed 7/5-7/7, comparing different drug dilutions at a certain
        timepoint across different types of cells. The measurements (whether they be cell toxicity or cell viability)
        are 'normalized' to the control measurements of that cell. The output of the function is a plot of this
        comparison.

    :param total_dict: the overall nested dictionary created from the plate reader output files (see
    read_all_plates_and_split() and build_dict_of_dicts(fp, param, conc_list)) - is a nested dictionary
    :param conc_list: the list of concentrations (drug dilutions) to be comparedthe cells are treated with across the
    desired cell types at the desired timepoint for the desired assay - should be a list of floats
    :param drug: the drug whose dilutions are being compared across cell types at a certain timepoint - should be a
    string
    :param timepoint: the timepoint of interest - should be a float
    :param cell_types: a list of the cell types to compare across differing dilutions of the drug of interest, at the
    correct timepoint; only cell types known to have this drug tested on them at the provided dilutions should be
    included - should be a list of strings
    :param assay: either 'lum' for the luminesence assay of cell viability, or 'Celltox' for the Celltox green assay of
    cell toxicity - is a string
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)

    '''

    # for testing:
    # cell_types = ['HEL', 'OCI', 'F36P', 'Kas3', 'NB4', 'SKM', 'MO16', 'PL21']

    ## To make the dataframe, will need a dictionary with the cell type as the key and as its value the flattened,
    #  ordered (by smallest->largest dilution) measurements (either viability or cell toxicity) as one list. The lists
    #  that will become the dictionary values must all be the same length, so first interrogate the number of replicates
    #  per cell type per drug dilution. The largest number of replicates found will be used as the number 'expected' for
    #  all cell types for all drug dilutions, to ensure list length equality.

    num_reps = []
    for i in cell_types:
        for j in total_dict[i][assay][timepoint][drug]:
            num_reps.append(len(total_dict[i][assay][timepoint][drug][j])) # make list of all replicate numbers
    num_rep = max(num_reps)

    # 'rep_list' is the list of replicates that will be used to index the data frame. Each index in rep_list corresponds
    # to one replicate plate-well measurement value. (If a cell type/drug treatment/drug dilution does not have enough
    # replicates, it will be handled.)
    rep_list = list(np.arange(num_rep))

    # 'concs_sorted_expanded' is the list of drug dilutions that will be used to index the dataframe along with
    # 'rep_list'. However, there are (usually) multiple replicates per drug dilution. Thus, each drug dilution will be
    # repeated for as many replicates as are tested at this dilution. The list is sorted to make sure the dilution in
    # the dataframe index matches the dilution used in the plate-well from which the measurement was taken.
    concs_sorted_expanded = sorted(conc_list * num_rep)

    ## Now make the dictionary with the cell type as the key and as the value the flattened, ordered list of
    #  measurements of cell type+drug at each dilution. Sort the keys so that the correct dilution's measurements are
    #  accessed from the overall nested dictionary, leading to correctly ordered measurements in the dataframe per cell
    #  type that match up with dilutions in the dataframe index.

    valuedict_from_sorted_keys = {}
    for i in cell_types:
        # values will come from the overall nested dictionary, so each cell type is only represented once
        valuedict_from_sorted_keys[i] = []
        for conc in sorted(list(set(concs_sorted_expanded))):
            # make sure the dilution in the list has been measured for this cell type/assay/timepoint/drug
            if conc in total_dict[i][assay][timepoint][drug]:
                valuedict_from_sorted_keys[i].extend(total_dict[i][assay][timepoint][drug][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as other cell types in
                # passed in, supplement the measurements it does have with 'None' values
                if not len(total_dict[i][assay][timepoint][drug][conc]) == num_rep:
                    valuedict_from_sorted_keys[i].extend([None] * (num_rep - len(total_dict[i][assay][timepoint][drug][conc])))
            # if the dilution in the list has not been measured for this cell type/assay/time/drug
            else:
                # give it as many 'None's as necessary to match up with other measured cell types
                # note - a cell type with all Nones will look weird in the resulting plot
                valuedict_from_sorted_keys[i].extend([None] * num_rep)

    ## Now make the dataframe, indexed both by dilutions (given in conc_list) and by replicates (determined during this
    #  function). For the concentration list given on 7/7/16, the index looks best as an x-axis when using the log (base
    #  10) of the concentrations - that way they're evenly spaced. Because of a bug in pandas/matplotlib, yerr can only
    #  be plotted when the index includes zero as one of its values, so I take the largest (in this case, least
    #  negative) value of the log10(concs) and subtract it from all log-concentration values, (in this case, subtracting
    #  a negative, thus adding that value). This means that log-concentration value is now zero, and all
    #  log-concentration values are shifted the same amount. (The correct log-concentration values will overwrite these
    #  incorrect values on the x-axis when the graph is plotted.)

    z = [math.log10(k * 10e-9) for k in concs_sorted_expanded] # drug dilutions per replicate, log (base 10) value
    # make the multiple index for the dataframe, giving the list of drug dilutions per replicate, and replicate list
    # multiplied by the number of concentrations - this way both lists are the same length and can be used properly to
    # index the dataframe
    ix = pd.MultiIndex.from_arrays([list(np.subtract(z, [max(z)] * len(z))), rep_list*len(conc_list)],
                                   names=['dilutions', 'replicates'])
    # make the dataframe using the index we just made, and give as the dictionary input required the dictionary of each
    # cell type and its flattened measurement values per replicate per dilution
    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## For this particular method we divide the plate-well measurement by the average of this measurement of the control
    #  cells for this cell type. The 'control' tuple gives the type of control ('None:x' for cells only, 'DMSO' for
    #  cells treated with DMSO, or 'Media-No_cells' for wells of only media) and the single dictionary key used by that
    #  type of control in the overall nested dict (cells only key is 0, DMSO key is 'N_A', media only key is 0 - see
    #  function 'read_drugs_table()')

    for i in df:
        df[i] = df[i] / np.mean(total_dict[i][assay][timepoint][control[0]][control[1]])

    ## Group the cells by dilutions, because we're going to take the mean of the replicates and leave the dilutions as
    #  they are (to make sure we get the mean for each one). The calculate the mean and standard deviation, to use for
    #  the plot.

    gp = df.groupby(level=('dilutions'))
    means = gp.mean()
    errors = gp.std()

    ## Make the list of strings to use for the x-axis - the nontransformed values with which we'll replace the
    #  transformed values (again, transformed values are a hack to get around the pandas/matplotlib problem with yerr).
    #  This problem should be fixed (according to pandas github) with a July 2016 pandas update, 1.18.2 - not released
    #  as of this writing.

    zz = [str(round(i, 2)) for i in sorted(list(set(z)))] # list of strings for correct x-axis labeling
    fig, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax, xticks=means.index)
    ax.get_xaxis().set_ticklabels(zz, rotation='vertical')
    ax.set_xlabel('log([Cmpd]), M')
    if assay == 'lum':
        ax.set_ylabel('Viability (% of control)')
    elif assay == 'Celltox':
        ax.set_ylabel('Cell toxicity (% of control)')
    ax.set_title(drug+' '+str(timepoint)+' hr')
    home = os.path.expanduser("~")
    fig5 = os.path.join(home, ('Drug_treatment_vs_control_'+drug+'_'+str(timepoint)+'_'+assay+'.png'))
    plt.savefig(fig5)
    plt.close()


def new_diff_ct_same_drug(total_dict, conc_list, drug, timepoint, cell_types, assay, control=('DMSO:2', 'N_A')):
    '''

        This function makes the type of plot discussed 7/5-7/7, comparing different drug dilutions at a certain
        timepoint across different types of cells. The measurements (whether they be cell toxicity or cell viability)
        are 'normalized' to the control measurements of that cell. The data is then fitted to a 4-parameter sigmoidal
        function, similar to to the LL.4 function from the R drc function. The output of the function is a plot of the
        data and the fitted function.

    :param total_dict: the overall nested dictionary created from the plate reader output files (see
    read_all_plates_and_split() and build_dict_of_dicts(fp, param, conc_list)) - is a nested dictionary
    :param conc_list: the list of concentrations (drug dilutions) to be comparedthe cells are treated with across the
    desired cell types at the desired timepoint for the desired assay - should be a list of floats
    :param drug: the drug whose dilutions are being compared across cell types at a certain timepoint - should be a
    string
    :param timepoint: the timepoint of interest - should be a float
    :param cell_types: a list of the cell types to compare across differing dilutions of the drug of interest, at the
    correct timepoint; only cell types known to have this drug tested on them at the provided dilutions should be
    included - should be a list of strings
    :param assay: either 'lum' for the luminesence assay of cell viability, or 'Celltox' for the Celltox green assay of
    cell toxicity - is a string
    :param control: a tuple denoting the appropriate key name in the nested dictionary for the desired control, and the
    appropriate key name for that control's single key with its measurement values - the first tuple index should be a
    string, and the second should be whatever is appropriate for that control (either a string, 'N_A', or a number, 0)

    '''

    #####
    ##
    ## The beginning of this function is identical to the beginning of old_diff_ct_same_drug(). For descriptions
    ## of what is occuring in the beginning of this function, refer to the comments in old_diff_ct_same_drug().
    ##

    num_reps = []
    for i in cell_types:
        for j in total_dict[i][assay][timepoint][drug]:
            num_reps.append(len(total_dict[i][assay][timepoint][drug][j])) # make list of all replicate numbers
    num_rep = max(num_reps)

    rep_list = list(np.arange(num_rep))

    concs_sorted_expanded = sorted(conc_list * num_rep)

    valuedict_from_sorted_keys = {}
    for i in cell_types:
        # values will come from the overall nested dictionary, so each cell type is only represented once
        valuedict_from_sorted_keys[i] = []
        for conc in sorted(list(set(concs_sorted_expanded))):
            # make sure the dilution in the list has been measured for this cell type/assay/timepoint/drug
            if conc in total_dict[i][assay][timepoint][drug]:
                valuedict_from_sorted_keys[i].extend(total_dict[i][assay][timepoint][drug][conc])
                # if this cell type/assay/timepoint/drug doesn't have as many replicates measured as other cell types in
                # passed in, supplement the measurements it does have with 'None' values
                if not len(total_dict[i][assay][timepoint][drug][conc]) == num_rep:
                    valuedict_from_sorted_keys[i].extend([None] * (num_rep - len(total_dict[i][assay][timepoint][drug][conc])))
            else:
                # give it as many 'None's as necessary to match up with other measured cell types
                # note - a cell type with all Nones will look weird in the resulting plot
                valuedict_from_sorted_keys[i].extend([None] * num_rep)

    ix = pd.MultiIndex.from_arrays([concs_sorted_expanded, [math.log10(k * 10e-9) for k in concs_sorted_expanded],
                                    rep_list * len(conc_list)],names=['dilutions', 'log dilutions', 'replicates'])
    df = pd.DataFrame(valuedict_from_sorted_keys, index=ix)

    ## This ends the code that is nearly identical to much of the code in old_diff_ct_same_drug(). The following code
    ## will have its own comments.
    ##
    #####

    ## Ideally we'll normalize the data by the cell count (/toxicity) at concentration "zero". Here, instead of using
    #  the control as our "drug concentration zero", we'll fit the drug concentration vs. cell viability (or drug
    #  concentration vs. cell toxicity) data to a curve. Since the x-axis (and x-values) denote(s) drug concentration,
    #  we'll look at the fitted curve at x=0 to get our "concentration zero" cell count / cell toxicity number. Then
    #  we'll normalize by that number.

    # we want a df that we won't add a zero to, for plotting purposes - we only want to show data points that were
    # experimentally determined, not the "concentration zero" point that we'll add from fitting a Hill curve
    df_plot = df.copy()

    # need to divide by an arbitrary number, close to the order of magnitude of most of the measurements leaving the
    # measurements in the thousands gives the curve_fit function trouble - probably rounding error during ll4() process
    DIV_NUM = 1000
    for i in df:
        df[i] = df[i] / DIV_NUM


    ## For each cell type, fit its assay values to a Hill curve and from that, determine a value at x=0 (concentration
    #  zero). Store the assay values for the "concentration zero" cell type to be added into the dataframe.

    conc_zeroes = {}
    for i in cell_types:
        # df.index.get_level_values() function returns the values in the specified index - the integers 0->whatever
        # specify your indices from left->right. DataFrame df has 3 indices - 'dilutions', 'log dilutions' and
        # 'replicates' (in that order) - so get_level_values(0) gives the values held in the 'dilutions' index
        popt, pcov = opt.curve_fit(ll4, df.index.get_level_values(0).values, df[i], maxfev=2000)
        conc_zero = ll4(0,*popt)
        conc_zeroes[i] = conc_zero

    ## For each cell type and its stored "concentration zero" value, make a DataFrame for a 'dilutions' (and 'log
    #  dilutions') value of 0 (plus replicate value of zero - meaningless) and the calculated assay-value-at-zero
    #  from the Hill curve fit. Then concatenate the DataFrames and move on to normalize by the appropriate number
    #  (the concentration zero value).

    nix = pd.MultiIndex.from_arrays([[0], [0], [0]], names=['dilutions', 'log dilutions', 'replicates'])
    df = pd.DataFrame(conc_zeroes,index=nix).append(df)

    # multiply the values back out by the arbitrary number from before - now the DataFrame has the original values (and
    # the proper "concentration zero" value
    for i in df:
        df[i] = df[i] * DIV_NUM

    ## Now, we can normalize the data (the experimental data, for plotting) by the values calculated by the fit function
    #  for 'concentration zero'.

    for i in df_plot:           # we know that df and df_plot have the same columns, because we haven't changed those
        df_plot[i] = df_plot[i]/df.iloc[0][i] # iloc[0] is the first row of the data frame - "concentration zero"

    ## Now we want to plot the data appropriately (experimental values only), and also calculate the IC50 and EC50
    #  values. Take the mean and standard deviation values of the replicates, to plot the mean for each concentration
    #  and error bars. Then for plotting, get the fit curves again and plot them along with data points and error bars.

    gp = df_plot.groupby(level=('dilutions', 'log dilutions'))
    means = gp.mean()
    errors = gp.std()

    for i in errors:
        errors.rename(columns={i: i + '-err'}, inplace=True)
    m_e = pd.concat([means, errors], axis=1, join_axes=[means.index])

    # The key to running the fit function is that you need to fit the curve on the non-log-transformed dilutions
    # (the x-vals in the function), because part of the function is calculating the log of the data. If you've already
    # taken them and passed them in, the ll4() function will be taking the logs of negative numbers which isn't
    # possible!
    # THEN, for plotting, plot with the logs as the x axis to view it appropriately

    for i, col in zip(cell_types, sns.color_palette("husl", len(cell_types))):
        # will give the same fit as before because passing in the same x-vals (we're not including the "conc zero" val)
        popt, pcov = opt.curve_fit(ll4, m_e.index.get_level_values(0).values, m_e[i],maxfev=2000)
        ec50 = str(round(math.log10(popt[3]*10e-9),2)) #ec50 is the 4th constant solved by ll4()
        ic50 = "None"
        # Interpolating the points and using that to find the concentration value at cell count/tox = .5 (by shifting
        # the curve down by .5 (since it starts at 1.0) and finding the zero) gave much more accurate values than any
        # other scipy.optimize or scipy.polynomial solvers I tried.
        st, sc, sk = scipy.interpolate.splrep(m_e.index.get_level_values(0).values,
                                          ll4(m_e.index.get_level_values(0).values, *popt))
        if len(scipy.interpolate.sproot((st, sc - .5, sk))) > 0:    # if no answer returned, then there's no ic50
            ic50 = str(round(math.log10(scipy.interpolate.sproot((st, sc - .5, sk))[0] * 10e-9),2))
            # get_level_values(0) gives the values held in the 'dilutions' index, get_level_values(1) in 'log dilutions'
        plt.errorbar(m_e.index.get_level_values(1).values, m_e[i], yerr=m_e[i + '-err'], fmt='o', color=col,
                         label=(i+'\n'+'IC50: '+ic50+'\n'+'EC50: '+ec50))
        plt.plot(m_e.index.get_level_values(1).values, ll4(m_e.index.get_level_values(0).values, *popt), color=col,
                 label=None)

    # Get graph axes, shrink the plot's width a little, then put a legend outside the plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([round(min(m_e.index.get_level_values(1).values)-.5),
              round(max(m_e.index.get_level_values(1).values)+.5)+.5])
    plt.xlabel('log([Cmpd]), M')
    if assay == 'lum':
        plt.ylabel('Viability (% of control)')
    elif assay == 'Celltox':
        plt.ylabel('Cell toxicity (% of control)')
    plt.title(drug+' '+str(timepoint)+' hr')
    # plt.savefig('New_Drug_treatment_vs_control_'+drug+'_'+str(timepoint)+'_'+assay+'.png')
    home = os.path.expanduser("~")
    fig = os.path.join(home, ('New_Drug_treatment_vs_control_' + drug + '_' + str(timepoint) + '_' + assay + '.png'))
    plt.savefig(fig)
    # print(fig)
    # plt.close()


def ll4(x,b,c,d,e):
    '''
    @yannabraham from Github Gist - https://gist.github.com/yannabraham/5f210fed773785d8b638

    This function is basically a copy of the LL.4 function from the R drc package with
     - b: hill slope
     - c: min response
     - d: max response
     - e: EC50
     '''
    return(c+(d-c)/(1+np.exp(b*(np.log(x)-np.log(e)))))


def cell_count_at_time_zero(ct_dict, time_list, control=('DMSO:2','N_A')):
    '''

        Assuming that N(t) = N(0)2**(t*rate) holds, this uses number of cells of the cell type for which ct_dict is
        the nested dict at t = 3 (3 days, or 72 hours) to determine the number of cells of that cell type at time t = 0.

        Calls determine_cell_growth_rate() to determine the growth rate (given in number of population doublings over time)
        that is required for this equation.

    :param ct_dict: information about the desired cell type as that cell type's subset of the overall nested dict -
    should be a nested dict
    :param time_list: the timepoints available for the desired cell type's cell counts (luminescence measurement) -
    should be a list of floats
    :param control: the appropriate control for the cell type of interest; will depend on the plate that the cell type
    came from... while drug of interest doesn't show up in this function, whatever called it is looking at a cell type
    in conjunction with one or more drugs. Whatever drug the cell type of interest is being treated with should be on
    the same plate as the control passed in here, so the measurements for its growth rate are from the same plate as the
    measurements for its drug treatment - should be a tuple denoting the appropriate key name in the nested dictionary
    for the desired control, and the appropriate key name for that control's single key with its measurement values;
    that is, the first tuple index should be a string, and the second should be whatever is appropriate for that control
    (either a string, 'N_A', or a number, 0)
    :return: number of cells of this cell type at time t=0
    '''

    # get the cell type's growth rate, for this particular plate (that means for the plate type that the cell was on in
    # conjunction with the drug of interest - this is passed in as the appropriate control tuple by the call to this
    # function)
    gr = determine_cell_growth_rate(ct_dict, time_list, control)

    # use 72 hours (3 days) as the time to use as t for N(t)
    # if 72 hours not available, use the next smallest time in the list
    time_t = 72
    if time_t not in time_list:
        for i in time_list:
            if i < 72:
                time_t = i
    N_t = np.mean(ct_dict['lum'][time_t][control[0]][control[1]])

    # from the above equation -> N(0) = N(t) / (2**(t*rate))
    N_0 = N_t / (2**((time_t/24)*gr))   # time_t / 24 because it needs to be in days

    return N_0


def determine_cell_growth_rate(ct_dict, time_list, control=('DMSO:2','N_A')):
    '''

    @ Darren Tyson

    When measurements are available for more than two time points, the rate of proliferation is obtained by
    least-squares regression of a linear model (log2(cell number) per time) where the slope of the linear model is
    the rate of proliferation in population doublings per unit time.

    :param ct_dict: information about the desired cell type as that cell type's subset of the overall nested dict -
    should be a nested dict
    :param time_list: the timepoints available for the desired cell type's cell counts (luminescence measurement) -
    should be a list of floats
    :param control: the appropriate control for the cell type of interest; will depend on the plate that the cell type
    came from... while drug of interest doesn't show up in this function, whatever called it is looking at a cell type
    in conjunction with one or more drugs. Whatever drug the cell type of interest is being treated with should be on
    the same plate as the control passed in here, so the measurements for its growth rate are from the same plate as the
    measurements for its drug treatment - should be a tuple denoting the appropriate key name in the nested dictionary
    for the desired control, and the appropriate key name for that control's single key with its measurement values;
    that is, the first tuple index should be a string, and the second should be whatever is appropriate for that control
    (either a string, 'N_A', or a number, 0)
    :return: the slope of the linear model denoting the rate of cell proliferation (in population doublings per unit
    time) - is a float
    '''

    # measuring time in days - will give population doublings per day
    t_as_days = [t/24 for t in time_list]
    log2_cell_counts = []
    for t in t_as_days:
        log2_cell_counts.append(math.log(np.mean(ct_dict['lum'][t*24][control[0]][control[1]]),2))
    # linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_as_days, log2_cell_counts)
    # slope is the rate of cell proliferation
    return slope


def run_example():

    ## To run this from outside you'll still need some arguments. As of now, the arguments are:
    #
    # - A dictionary whose keys are 'plate type' numbers, following the -1 / -2 / -3 / -4 convention discussed in our
    #   7/7/16 meeting; values are the 9 drugs (in order) used on those plates
    #       -- For instance, in the printout plate layout sheets from 7/7/16 meeting, plate type -2 (string "2" as the
    #          dictionary key - see 'param' below - has drugs in order 1 through 9, and these are listed in that order
    #          as the "2" key's values in the dictionary. Again, see 'param' below (in conjunction with 7/7/16 meeting
    #          plate layout sheet).
    #
    # - A list of drug concentrations, in the order they're put on the plate. The concentrations are expected in
    #   nanomolar, for example a value of 10uM should be input as 10,000 in the list of drug concentrations. The
    #   concentrations will be converted to molar (= multiplied by 10e-9) in each plotting function.
    #       -- The "order" they're put on the plate is the order, left to right, in which the wells of each drug is
    #          treated. This follows the legend on the 7/7/16 meeting plate layout printout (top right).
    #
    # - The path to the directory where the plate files are stored.
    #
    # - The path to the directory where the split-up plate files should be kept, and where they'll be read from by the
    #   function that makes the nested dictionary of plate data.
    #       -- I considered making this directory temporary, i.e. as soon as the nested dictionary is made, the
    #          directory (and all files within) is removed. This should be a trivial change, should someone want to
    #          do it.
    #
    # - Some way to indicate which plots should be created, AND THEN
    # - Some way to indicate what values (cell type(s)/drug(s)/timepoint(s)/assay(s)) those plots should look at
    #       -- These 2 haven't been set up or prepared for in this code, so that will take some extra effort. Sorry!
    #
    ##

    ## Instead of taking input arguments, here I hardcode the dict of plate types and drugs on each plate, the
    #  list of concentrations in the order the wells are treated, and the directories where the plate files can be found
    #  and where the split-up plate files will be written to (and then read from to make the nested dict of data).

    param = {'2': ['24360 IDO', '39110 JAK1', '52793 JAK1', '53914 PIM', '50465 PI3Kd', '54828 FGFR', '54329 BRD',
                   'Decitabine', 'Cytarabine'],
             '3': ['SAHA', 'ATRA', 'Imatinib', 'Ruxolitinib', 'ABT-199', 'Daunorubicin', 'VU0661013-5 ("013)',
                   'VU0807260-1 ("013 neg")', 'R-SER-30'],
             '4': ['JAK1 + 10nM ATRA', 'JAK1 + 100nM ATRA', 'JAK1 + 1uM ATRA', 'JAK1 + 10nM BRD', 'JAK1 + 30 nM BRD',
                   'JAK1 + 100nM BRD', 'BRD + 10nM Imatinib', 'BRD + 30nM Imatinib', 'BRD + 100nM Imatinib']}
    conc_list = [10000, 3000, 1000, 300, 100, 30, 10, 3, 1, .3] # these vals will be multipled by 10e-9 before plotting
    home = os.path.expanduser("~")
    path_to_plate_files = os.path.join(home,'hts_plate_files')
    path_to_split_files = os.path.join(home,'hts_split_files')

    ## Now all the functions I wrote are called. The first are leading up to the creation of 'd', the nested dict of all
    #  the plate data. Then, I recreated (or in one case made the same type of plot as) the plots in the HTS Data for
    #  Lopez Lab Dropbox. Finally, I wrote some new functions - one with a different way of normalizing and IC50 and
    #  EC50 calculations, one to look at cell counts over time. This last one will help set up the scene for someone to
    #  code in DIP calculation on the data.

    fp = read_all_plates_and_split(path_to_plate_files,path_to_split_files)
    d = build_dict_of_dicts(fp, param, conc_list)
    # d is the nested dict used to make the plots - it has all the cell count / cell tox information from the plates

    # 'Recreations' of HTS Data for Lopez Lab, example HTS figures slides - loosely used 'recreations' here because I
    # didn't use a fit function for this data. Mainly to see if our numbers are correct. To add a fit function, consult
    # the code in new_diff_ct_same_drug() and use the curve_fit() and ll4() calls there to guide you.

    # recreates the dropbox image on the left of slide 4 of 16-07-07_Example HTS Figures.pptx

    old_diff_ct_same_drug(d, conc_list, '54329 BRD', 96,
                          ['HEL', 'OCI', 'F36P', 'Kas3', 'NB4', 'SKM', 'MO16', 'PL21'], 'lum', ('DMSO:2', 'N_A'))
    # recreates the dropbox image on the right of slide 4 of 16-07-07_Example HTS Figures.pptx
    old_diff_ct_same_drug(d, conc_list, '54329 BRD', 96,
                          ['HEL', 'OCI', 'F36P', 'Kas3', 'NB4', 'SKM', 'MO16', 'PL21'], 'Celltox', ('DMSO:2', 'N_A'))

    # is the same type of graph as the dropbox image on slide 5 of 16-07-07_Example HTS Figures.pptx, but Kas1 data
    # not included in the Dropbox
    drug_list = ['54329 BRD', 'BRD + 10nM Imatinib', 'BRD + 30nM Imatinib', 'BRD + 100nM Imatinib']
    old_same_ct_diff_drug(d['OCI'], 'OCI', conc_list, drug_list, 96, 'lum', ('DMSO:4', 'N_A'))

    # recreates the dropbox image left upper part of slide 6 of 16-07-07_Example HTS Figures.pptx
    old_same_ct_drug_diff_time(d['F36P'], 'F36P', conc_list, '52793 JAK1', [24,48,72,96], 'lum', ('DMSO:2', 'N_A'))
    # recreates the dropbox image left lower part of slide 6 of 16-07-07_Example HTS Figures.pptx
    old_same_ct_drug_diff_time(d['F36P'], 'F36P', conc_list, '52793 JAK1', [24,48,72,96], 'Celltox', ('DMSO:2', 'N_A'))
    # recreates the dropbox image right upper part of slide 6 of 16-07-07_Example HTS Figures.pptx
    old_same_ct_drug_diff_time(d['Kas3'], 'Kas3', conc_list, '52793 JAK1', [24,48,72,96], 'lum', ('DMSO:2', 'N_A'))
    # recreates the dropbox image right lower part of slide 6 of 16-07-07_Example HTS Figures.pptx
    old_same_ct_drug_diff_time(d['Kas3'], 'Kas3', conc_list, '52793 JAK1', [24,48,72,96], 'Celltox', ('DMSO:2', 'N_A'))

    # ALMOST recreates the dropbox image on the left of slide 7 of 16-07-07_Example HTS Figures.pptx
    # difference here: no implementation of double y-axes (I'm sure this is possible)
    old_viab_vs_celltox(d['F36P'], 'F36P', conc_list, '52793 JAK1', 48, control=('DMSO:2', 'N_A'))
    # ALMOST recreates the dropbox image on the right of slide 7 of 16-07-07_Example HTS Figures.pptx
    # differnce here: no implementation of double y-axes (I'm sure this is possible)
    old_viab_vs_celltox(d['F36P'], 'F36P', conc_list, '52793 JAK1', 96, control=('DMSO:2', 'N_A'))


    # New functions
    new_diff_ct_same_drug(d, conc_list, '54329 BRD', 96,
                          ['HEL', 'OCI', 'F36P', 'Kas3', 'NB4', 'SKM', 'MO16', 'PL21'], 'lum',
                          control=('DMSO:2', 'N_A'))

    new_cells_over_time(d['F36P'], 'F36P', conc_list, '52793 JAK1', [24,48,72,96], 'lum', control=('DMSO:2', 'N_A'))

if __name__ == '__main__':
    run_example()
