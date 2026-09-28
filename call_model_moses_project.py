# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:26:39 2022

@author: arvensyla
"""

import sys, os, socket

import numpy as np
import pandas as pd
from importlib import reload

import grimsel.grimsel_config as config
import pyomo.environ as po

import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB
from grimsel.core.io import IO as IO
from grimsel.auxiliary.aux_m_func import set_to_list

import grimsel.core.io as io
import model_loop_modifier_moses_project as model_loop_modifier
# import model_loop_modifier_ev_res_ev_others_v2g_hp_cooling_as_dmnd_tsp as model_loop_modifier
# from pyAndy import PlotPageData, PlotTiled

import logging

logger = logging.getLogger('grimsel')
logger.setLevel(logging.INFO)

#sc_out = os.path.abspath('C:/Users/sylaarv1/OneDrive - Université de Genève/Desktop/Work/MOSES Project - Fourth paper/GRIMSEL_moses/grimsel_moses/grimsel-master-training/xxx1')
sc_out = os.path.abspath('/home/users/s/syla/output_files/MOSES_project/grimsel_artelys_scenario_EMR12_EMR12plus_noNUC_LoL')
# %%
IO._close_all_hdf_connections()

reload(model_loop)
reload(model_loop_modifier)
reload(config)

# ~~~~~ ADD nd_weight TO DEF_NODE
path_dfnd = os.path.join(config.PATH_CSV, 'def_node.csv')
df_def_node = pd.read_csv(path_dfnd)
df_def_node = df_def_node[[c for c in df_def_node.columns if not 'Unnamed' in c]]

df_def_node['nd_weight'] = 1

df_def_node.to_csv(path_dfnd, index=False)


# ~~~~~ Fix data types for node_encar
path_dfne = os.path.join(config.PATH_CSV, 'node_encar.csv')
node_encar = pd.read_csv(path_dfne)
node_encar = node_encar[[c for c in node_encar.columns if not 'Unnamed' in c]]

node_encar.to_csv(path_dfne, index=False)

#%%

# ~~~~~ Fix data types for plant_encar 

path_dfnd_pc = os.path.join(config.PATH_CSV, 'plant_encar.csv')
df_plant_encar = pd.read_csv(path_dfnd_pc)
df_plant_encar['ca_id'] = df_plant_encar['ca_id'].astype(int) 
# df_plant_encar.loc[df_plant_encar['pp_id'] == 39, 'cap_pwr_leg_yr2050'] = 29690
df_plant_encar['cap_pwr_leg_yr2050'] = df_plant_encar['cap_pwr_leg_yr2050'].astype(float) 
# df_plant_encar.loc[df_plant_encar['pp_id'] == 19, ['cap_pwr_leg_yr2030','cap_pwr_leg_yr2035']] = 9350.95

path_dfnd_dp = os.path.join(config.PATH_CSV, 'def_plant.csv')
df_def_plant = pd.read_csv(path_dfnd_dp)
battery_list = df_def_plant[df_def_plant['pp'].str.contains('DE_STO_LI|IT_STO_LI|FR_STO_LI|AT_STO_LI')].pp_id.to_list()

df_plant_encar[df_plant_encar.pp_id.isin(battery_list)]
df_plant_encar.loc[df_plant_encar['pp_id'].isin(battery_list), 'vc_om'] = 1
df_plant_encar.loc[df_plant_encar['pp_id'].isin(battery_list), 'discharge_duration'] = 4
df_plant_encar.loc[df_plant_encar['pp_id'].isin(battery_list), 'cap_pwr_leg_old'] = 0
df_plant_encar.loc[df_plant_encar['pp_id'].isin(battery_list), ['fc_om_yr2020', 'fc_om_yr2025', 'fc_om_yr2030', 'fc_om_yr2035', 'fc_om_yr2040', 'fc_om_yr2045', 'fc_om_yr2050']] = 0

battery_list_archetypes = df_def_plant[df_def_plant.pp.str.contains('STO_LI|VR') & ~df_def_plant.pp.str.contains('DE_STO_LI|IT_STO_LI|FR_STO_LI|AT_STO_LI')].pp_id.to_list()
df_plant_encar.loc[df_plant_encar['pp_id'].isin(battery_list_archetypes), 'vc_om'] = 0.5


hp_dhw_pp_eff_list = df_def_plant[df_def_plant['pp'].str.contains('DHW_AW|HP_AW')].pp_id.to_list()
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_pp_eff_list), 'pp_eff'] = 1
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_pp_eff_list), ['discharge_duration', 'st_lss_rt', 'factor_lin_0', 'factor_lin_1']] = 0
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_pp_eff_list), 'cap_pwr_leg'] = 1
hp_dhw_sto_list = df_def_plant[df_def_plant['pp'].str.contains('DHW_HP_STO')].pp_id.to_list()
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_sto_list), ['st_lss_rt', 'factor_lin_0', 'factor_lin_1']] = 0
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_sto_list), 'cap_pwr_leg'] = 1
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_dhw_sto_list), 'discharge_duration'] = 0

hp_sto_list = df_def_plant[df_def_plant['pp'].str.contains('STO_HT')].pp_id.to_list()
df_plant_encar.loc[df_plant_encar['pp_id'].isin(hp_sto_list), 'cap_pwr_leg'] = 1


pv_arch_list = df_def_plant[df_def_plant['pp'].str.contains('_PHO') & ~df_def_plant['pp'].str.contains('DE_SOL|IT_SOL|FR_SOL|AT_SOL|CH_SOL')
].pp_id.to_list()

df_def_plant[df_def_plant.pp_id.isin(pv_arch_list)]
df_def_plant.loc[df_def_plant['pp_id'].isin(pv_arch_list), 'set_def_add'] = 0

#cols = ['fc_cp_ann_yr2020','fc_cp_ann_yr2025','fc_cp_ann_yr2030','fc_cp_ann_yr2035']

#df_plant_encar.loc[df_plant_encar['pp_id'].isin(pv_arch_list), cols] *= 0.7

df_plant_encar.to_csv(path_dfnd_pc, index=False)

df_def_plant.to_csv(path_dfnd_dp, index=False)

# ~~~~~
path_coppr = os.path.join(config.PATH_CSV, 'def_cop_35.csv')
df_def_cop_35 = pd.read_csv(path_coppr)
df_def_cop_35['DateTime'] = pd.to_datetime(df_def_cop_35.DateTime)

path_coppr = os.path.join(config.PATH_CSV, 'def_cop_60.csv')
df_def_cop_60 = pd.read_csv(path_coppr)
df_def_cop_60['DateTime'] = pd.to_datetime(df_def_cop_60.DateTime)

path_coppr = os.path.join(config.PATH_CSV, 'def_cop_60_dhw.csv')
df_def_cop_60_dhw = pd.read_csv(path_coppr)
df_def_cop_60_dhw['DateTime'] = pd.to_datetime(df_def_cop_60_dhw.DateTime)

df_def_cop_mean = pd.merge(df_def_cop_35,df_def_cop_60,on=['DateTime','doy','pp_id','hy']).assign(
        value = lambda x: (x.value_x+x.value_y)/2)
# ~~~~~
# ~~~~~

excl_pt = [
    #  '%PRC%', '%SLL%',
    # # '%%STO%SFH',
    #   '%PHO',
    # '%PHO_SFH%',
    # '%GAS_NEW%',
    #   '%GAS_LIN%',
   #   '%NUC%', '%HCO%', '%LIG%',
    # # '%LOL%',
    #     '%HYD_RES%',
    #     '%HYD_ST%',
    #     '%HYD_ROR%',
    #  '%OIL%', '%GEO%', '%BAL%', '%BIO%', '%WAS%', '%WIN%',
    # # '%CAES%',
    # '%STO_LI%',
    # '%STO_VR%',
    # '%DMND_FLE%',
    '%HP_AW%',
    '%HP_WW%',
    # '%SFH%', '%MFH%', 
    # '%OCO%', '%IND%',
    '%STO_HT%',         
  #   '%BO%',             # DHW boilers
        # '%DHW_HP_STO%',        # DHW storgae from HP (buffer)
    # '%DHW_AW%',         # DHW for AW HP
    '%DHW_WW%',         # DHW for BW HP
]

if excl_pt == []:
    slct_pt = pd.read_csv(os.path.join(config.PATH_CSV, 'def_pp_type.csv')).pt.tolist()
else:
    slct_pt = pd.read_csv(os.path.join(config.PATH_CSV, 'def_pp_type.csv'))
    slct_pt = slct_pt.loc[
        np.invert(slct_pt.pt.str.contains('|'.join([pt.replace('%', '') for pt in excl_pt])))].pt.tolist()
slct_pt
# %%
nd_nt = df_def_node.loc[df_def_node.nd.isin([ 'AT0', 'IT0', 'FR0', 'DE0'])].nd_id.tolist()
nd_ch0 = df_def_node.loc[df_def_node.nd.isin(['CH0'])].nd_id.tolist()
nd_nt_ch0 = nd_nt + nd_ch0
nd_arch = df_def_node.loc[np.invert(df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0','Public_Charging']))].nd_id.tolist()
nd_arch_cl = df_def_node.loc[df_def_node.nd.str.contains('CL')].nd_id.tolist()
nd_arch_ht = df_def_node.loc[df_def_node.nd.str.contains('HT')].nd_id.tolist()
nd_arch_dsr = df_def_node.loc[df_def_node.nd.str.contains('DSR')].nd_id.tolist()
nd_arch_ev_res = df_def_node.loc[(df_def_node.nd.str.contains('EV')) & (df_def_node.nd.str.contains('SFH|MFH'))].nd_id.tolist()
nd_arch_ev_ind_oco = df_def_node.loc[(df_def_node.nd.str.contains('EV')) & (df_def_node.nd.str.contains('IND|OCO'))].nd_id.tolist()
nd_public_charging = df_def_node.loc[df_def_node.nd.str.contains('Public_Charging')].nd_id.tolist()

nd_arch_el = list(set(nd_arch) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev_res) - set(nd_arch_ev_ind_oco) - set(nd_arch_cl))

nd_all_str = df_def_node.nd.tolist()

nd_arch_el_res = df_def_node.loc[df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
nd_arch_el_res = list(set(nd_arch_el_res) - set(nd_arch_ev_res) - set(nd_arch_ht) - set(nd_arch_cl) - set(nd_arch_dsr))
nd_arch_ch = nd_ch0 + nd_arch


# Below, definition of nodes when wanting to choose timeframe selection when to satisfy the demand.
####
nd_ch0_str = df_def_node.loc[df_def_node.nd.isin(['CH0'])].nd.tolist()

nd_arch_ev_res_str = df_def_node.loc[(df_def_node.nd.str.contains('EV')) & (df_def_node.nd.str.contains('SFH|MFH'))].nd.tolist()
nd_arch_ev_ind_oco_str = df_def_node.loc[(df_def_node.nd.str.contains('EV')) & (df_def_node.nd.str.contains('IND|OCO'))].nd.tolist()
nd_public_charging_str = df_def_node.loc[df_def_node.nd.str.contains('Public_Charging')].nd.tolist()

nd_arch_cl_str = df_def_node.loc[df_def_node.nd.str.contains('CL')].nd.tolist()
nd_arch_ht_str = df_def_node.loc[df_def_node.nd.str.contains('HT')].nd.tolist()
nd_arch_dsr_str = df_def_node.loc[df_def_node.nd.str.contains('DSR')].nd.tolist()
nd_arch_not_ht_str = df_def_node.loc[np.invert(df_def_node.nd.str.contains('HT'))].nd.tolist()
nd_all_str = df_def_node.nd.tolist()
nd_arch_el_res_str = df_def_node.loc[(df_def_node.nd.str.contains('SFH|MFH'))  &~(df_def_node.nd.str.contains('CL')) &~(df_def_node.nd.str.contains('EV')) &~(df_def_node.nd.str.contains('HT')) &~(df_def_node.nd.str.contains('DSR'))].nd.tolist()


nd_all_wo_dsr_str = list(set(nd_all_str) - set(nd_arch_dsr_str))
path_dfplant = os.path.join(config.PATH_CSV, 'def_plant.csv')
df_def_plant = pd.read_csv(path_dfplant)
pp_hp_aw = df_def_plant.loc[df_def_plant.pp.str.contains('HP_AW')].pp_id.tolist()
pp_hp_ww = df_def_plant.loc[df_def_plant.pp.str.contains('HP_WW')].pp_id.tolist()

pp_dhw_aw = df_def_plant.loc[df_def_plant.pp.str.contains('DHW_AW')].pp_id.tolist()

# nhours_dict_ht = {nd : (24,24) for nd in nd_arch_ht_str} 

nhours_dict_cl = {nd : (1,1) for nd in nd_arch_cl_str} 
nhours_dict_ht = {nd : (1,1) for nd in nd_arch_ht_str} 
nhours_dict_dsr = {nd: (24,24) for nd in nd_arch_dsr_str}
nhours_dict_ev_res = {nd: (1,1) for nd in nd_arch_ev_res_str}
nhours_dict_ev_ind_oco = {nd: (1,1) for nd in nd_arch_ev_ind_oco_str}
nhours_dict_public_charging = {nd: (1,1) for nd in nd_public_charging_str}


# nhours_dict = dict(nhours_dict_el,**nhours_dict_ht)


nhours_dict = dict(nhours_dict_dsr,
                   **nhours_dict_cl,
                    **nhours_dict_ht,
                    **nhours_dict_ev_res,
                    **nhours_dict_ev_ind_oco,
                    **nhours_dict_public_charging,
                    )

# nhours_dict = dict(nhours_dict_el,**nhours_dict_ht)
# nhours_dict = nhours_dict_ht
####

# additional kwargs for the model
mkwargs = {
    'slct_encar':  # ['EL'],
                # ['EL','AW','HW','HW'],
                # ['EL','AW','WW','HW','HA','HB'],
                # ['EL','AW','HW','HA'],
                ['EL','HW','HA'],
                # ['EL','AW','HW', 'HA'],
            # ['EL'],
    'slct_node':
    # ['FR0', 'DE0'],
    # nd_all_str,
    # nd_all_wo_dsr_str,
        [
#                        'CH0', # 'DE0',
 #                       'MFH_RUR_0', # 'SFH_URB_HT_0', 'SFH_URB_0_DSR','SFH_URB_0_EV',
                     #  'MFH_URB_0','MFH_URB_HT_0', 'MFH_URB_0_DSR','MFH_URB_0_EV',
                #        'MFH_RUR_CL_0', # 'MFH_URB_HT_0_HP_AW',
  #                      'MFH_RUR_HT_0',
                     #  'SFH_URB_HT_0_STO_HT','MFH_URB_HT_0_STO_HT',
                     #  'SFH_URB_HT_0_DHW_AW', 'MFH_URB_HT_0_DHW_AW',
                     #  'SFH_URB_HT_0_HP_STO','MFH_URB_HT_0_HP_STO',
   #                     'MFH_RUR_0_EV',
    #                    'MFH_RUR_0_DSR',

                    #    'SFH_RUR_0', # 'SFH_URB_HT_0', 'SFH_URB_0_DSR','SFH_URB_0_EV',
                     #  'MFH_URB_0','MFH_URB_HT_0', 'MFH_URB_0_DSR','MFH_URB_0_EV',
                     #   'SFH_RUR_CL_0', # 'MFH_URB_HT_0_HP_AW',
                      #  'SFH_RUR_HT_0',
                     #  'SFH_URB_HT_0_STO_HT','MFH_URB_HT_0_STO_HT',
                     #  'SFH_URB_HT_0_DHW_AW', 'MFH_URB_HT_0_DHW_AW',
                     #  'SFH_URB_HT_0_HP_STO','MFH_URB_HT_0_HP_STO',
                   #     'SFH_RUR_0_EV',
                    #    'SFH_RUR_0_DSR',
                     #   'IND_RUR',
                      #  'OCO_RUR',
                     #   'SFH_SUB_0','SFH_SUB_HT_0', 'SFH_SUB_0_DSR','SFH_SUB_0_EV',
                     # 'MFH_SUB_0','MFH_SUB_HT_0', 'MFH_SUB_0_DSR','MFH_SUB_0_EV',
                     #  'SFH_URB_HT_0_HP_AW', # 'MFH_SUB_HT_0_HP_AW',
                     # 'SFH_URB_HT_0_STO_HT',
                     # 'SFH_URB_HT_0_DHW_AW',
                     # 'SFH_URB_HT_0_DHW_HP_STO',
                     # 'SFH_URB_0_PHO',
                     # 'SFH_URB_0_STO_LI',
                     
                        # 'SFH_SUB_HT_0_STO_HT','MFH_SUB_HT_0_STO_HT',
                     # 'SFH_SUB_HT_0_DHW_AW','MFH_SUB_HT_0_DHW_AW',
                     # 'SFH_SUB_HT_0_HP_STO','MFH_SUB_HT_0_HP_STO',

                     
        #               'SFH_RUR_0','SFH_RUR_HT_0', 'SFH_RUR_0_DSR','SFH_RUR_0_EV',
                      # 'MFH_RUR_0', 'MFH_RUR_HT_0', 'MFH_RUR_0_DSR','MFH_RUR_0_EV',
                       # 'MFH_RUR_0_PHO','MFH_RUR_0_STO_LI',
                     
        #             'IND_RUR', 'IND_SUB', 'IND_URB',
        #             'IND_RUR_EV', 'IND_SUB_EV', 'IND_URB_EV',
                    
                    # 'OCO_RUR', 'OCO_SUB', 'OCO_URB',
                    # 'OCO_RUR_EV', 'OCO_SUB_EV', 'OCO_URB_EV',
                      # 'Public_Charging',
        #             'AT0', 'IT0', 'FR0', 'DE0'
                            
                      # 'SFH_RUR_0', 'SFH_SUB_0', 
                      # 'SFH_URB_0_PHO', #'SFH_RUR_0_PHO', 
                      # 'SFH_SUB_0_PHO',
                      # 'SFH_URB_0_STO_LI', # 'SFH_RUR_0_STO_LI',  
                      # 'SFH_SUB_0_STO_LI',               
        #             # 'SFH_URB_1', 'SFH_RUR_1', 'SFH_SUB_1', 
        #             # 'SFH_URB_1_PHO', 'SFH_RUR_1_PHO', 'SFH_SUB_1_PHO',                    
        #             # 'SFH_URB_1_STO_LI', 'SFH_RUR_1_STO_LI', 'SFH_SUB_1_STO_LI',
        #             # 'SFH_URB_1', 'SFH_RUR_1', 'SFH_SUB_1',
        #              # 'SFH_URB_2', 'SFH_RUR_2', 'SFH_SUB_2',
        #              # 'SFH_URB_3', 'SFH_RUR_3', 'SFH_SUB_3',
        #            # 'SFH_URB_HT_0',# 'SFH_RUR_HT_0', 'SFH_SUB_HT_0',

                      # 'MFH_URB_0_PHO', # 'MFH_RUR_0_PHO', 
                      # 'MFH_SUB_0_PHO',
                      # 'MFH_URB_0_STO_LI', # 'MFH_RUR_0_STO_LI', 
                      # 'MFH_SUB_0_STO_LI',                  
        #             # 'MFH_URB_1', 'MFH_RUR_1', 'MFH_SUB_1',
        #             # 'MFH_URB_1_PHO', 'MFH_RUR_1_PHO', 'MFH_SUB_1_PHO',
        #             # 'MFH_URB_1_STO_LI', 'MFH_RUR_1_STO_LI', 'MFH_SUB_1_STO_LI',
        #              # 'MFH_URB_1', 'MFH_RUR_1', 'MFH_SUB_1',
        #             # 'MFH_URB_2', 'MFH_RUR_2', 'MFH_SUB_2',
        #             # 'MFH_URB_3', 'MFH_RUR_3', 'MFH_SUB_3',
                    
        #            # 'SFH_URB_0_EV', #'SFH_RUR_0_EV', 'SFH_SUB_0_EV',
        #             # 'SFH_URB_1_EV', 'SFH_RUR_1_EV', 'SFH_SUB_1_EV',                   
        #             # 'MFH_URB_0_EV', 'MFH_RUR_0_EV', 'MFH_SUB_0_EV',
        #             # 'MFH_URB_1_EV', 'MFH_RUR_1_EV', 'MFH_SUB_1_EV',

        #            # 'SFH_URB_0_V2G', #'SFH_RUR_0_V2G', 'SFH_SUB_0_V2G',
        #             # 'SFH_URB_1_V2G', 'SFH_RUR_1_V2G', 'SFH_SUB_1_V2G',                   
        #             # 'MFH_URB_0_V2G', 'MFH_RUR_0_V2G', 'MFH_SUB_0_V2G',
        #             # 'MFH_URB_1_V2G', 'MFH_RUR_1_V2G', 'MFH_SUB_1_V2G',
        
        #             # 'IND_RUR', 'IND_SUB', 
        #            # 'IND_URB',
        #             # 'IND_URB_PHO', 'IND_RUR_PHO', 'IND_SUB_PHO',
        #             # 'IND_URB_STO_LI', 'IND_RUR_STO_LI', 'IND_SUB_STO_LI',
        #             # 'OCO_RUR', 'OCO_SUB', 'OCO_URB',
        #             # 'OCO_URB_PHO', 'OCO_RUR_PHO', 'OCO_SUB_PHO',
        #             # 'OCO_URB_STO_LI', 'OCO_RUR_STO_LI', 'OCO_SUB_STO_LI',
                                      
                    ],
    #                            'IND_RUR',
    # ##
    'nhours': 
        nhours_dict,
        # 1,

    'slct_pp_type': slct_pt,
    #           'skip_runs': True,    
    'tm_filt': [
         # ('mt_id', [5]),
                                    # ('wk_id', [51]),
            # ('day', [1,4]),                       
                             
     #    ('doy', [4,5,6,7]),
#        ('doy', [4]), #173
        			 # ('hour', [24]),
        #                       ('hom', [0]),
                                # ('hour', range(1,25,6))
                               
        #                   ('hom', range(24))
    ],
    'symbolic_solver_labels': False,  # Set to True for debugging (set name to pyomo indices in cplex files)
    'constraint_groups':
        MB.get_constraint_groups(excl=['supply']),  # BE CAREFUL WITH THIS
    #               'chp','chp_new']),
    'nthreads': 2,  # Only passed to CPLEX
    'keepfiles': False,  # Set to True to keep tmp files
}

# additional kwargs for the i/o
iokwargs = {'sc_warmstart': False,
            'cl_out': sc_out,
            'resume_loop': False,
            #            'data_path': PATH_CSV,
            'no_output': False,
            'replace_runs_if_exist': False,
            'autocomplete_curtailment': False,
            'sql_connector': None,
            'data_path': config.PATH_CSV,
            'output_target': 'fastparquet',  # 'fastparquet',#'hdf5',#
            'dev_mode': True
            }

nsteps_default = [('swfy',5, np.arange),  # future years
                  ('swch', 2, np.arange),  # swiss scenarios
                  ('swhp', 1, np.arange),    # heatpumps scenarios
                  ('swtr', 1, np.arange),    # transmissions scenarios
                  ('swntc', 3, np.arange),    # NTC transmissions scenarios
                  ('swrf', 1, np.arange),    # retrofit scenario
                  ('swcl', 1, np.arange),    # cooling scenario
                  # ('swtsp', 1, np.arange),    # tempsetpoint scenario
                  ('swdpf', 1, np.arange),    # original, EE loads
                  ('swev', 1, np.arange),   # EV scenarios
                  ('swv2g', 1, np.arange),   # V2G scenarios                  
                  ]

mlkwargs = {  # 'sc_inp': 'lp_input_levels',
    #            'db': db,
    'nsteps': nsteps_default,
}

ml = model_loop.ModelLoop(**mlkwargs, mkwargs=mkwargs, iokwargs=iokwargs)
self = ml.m
ml.init_run_table()
# %

ml.io.read_model_data()

# %
ml.m.init_maps()

ml.m.map_to_time_res()

# %
ml.io.write_runtime_tables()

# ml.m.build_model()
# %


ml.m.get_setlst()
ml.m.define_sets()

ml.m.add_parameters()

ml.m.define_variables()
# %

ml.m.add_all_constraints()

#%%
#MB.delete_component(self,comp_name='supply')

df_cop_35 = df_def_cop_mean.loc[df_def_cop_mean.pp_id.isin([i for i in self.pp])]

df_cop_60_dhw = df_def_cop_60_dhw.loc[df_def_cop_60_dhw.pp_id.isin([i for i in self.pp])]

list_pp_cop = df_cop_35.pp_id.unique()
# list_pp_hp = self.df_def_plant.loc[df_def_plant.pp.str.contains('HP')].pp_id.to_list()
list_pp_hp = self.df_def_plant.loc[df_def_plant.pp.str.contains('HP_AW')].pp_id.to_list() # -> for 3rd paper
list_pp_hp_eff =  list(set(list_pp_hp) - set(list_pp_cop))
list_pp_cop = list(set(list_pp_hp) - set(list_pp_hp_eff))

list_pp_dhw_cop = df_cop_60_dhw.pp_id.unique()
list_pp_hp_dhw = self.df_def_plant.loc[df_def_plant.pp.str.contains('DHW_AW|DHW_WW')].pp_id.to_list()
list_pp_hp_dhw_eff =  list(set(list_pp_hp_dhw) - set(list_pp_dhw_cop))
list_pp_dhw_cop = list(set(list_pp_hp_dhw) - set(list_pp_hp_dhw_eff))

list_pp_dhw_bo = self.df_def_plant.loc[df_def_plant.pp.str.contains('DHW_BO')].pp_id.to_list()

# nd_arch_ht = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('HT')].nd_id.unique().tolist())

# if nd_arch_ht:
#     tm_id_ht = self.dict_nd_tm_id[nd_arch_ht[0]]
# else:
#     pass

#def add_supply_rules(self):
###############
def get_transmission(sy, nd, nd_2, ca, export=True):
            '''
            If called by supply rule, the order nd, nd_2 is always the ndcnn
            order, therefore also trm order.

            * **Case 1**: ``nd`` has higher time resolution (min) |rarr| just
              use ``trm[tm, sy, nd, nd_2, ca]``
            * **Case 2**: ``nd`` has lower time resolution (not min) |rarr|
              average ``avg(trm[tm, sy_2, nd, nd_2, ca])`` for all ``sy_2``
              defined by the
                ``grimsel.core.model_base.ModelBase.dict_sysy[nd, nd_2, sy]``

            Parameters
            ----------
            sy : int
                current time slot in nd
            nd : int
                outgoing node
            nd_2 : int
                incoming node
            ca : int
                energy carrier
            export : bool
                True if export else False

            '''
            if self.is_min_node[(nd if export else nd_2,
                                  nd_2 if export else nd)]:
                trm = self.trm[sy, nd, nd_2, ca]
                return trm

            else: # average over all of the other sy
                list_sy2 = self.dict_sysy[nd if export else nd_2,
                                          nd_2 if export else nd, sy]

                avg = 1/len(list_sy2) * sum(self.trm[_sy, nd, nd_2, ca]
                                            for _sy in list_sy2)
                return avg


def supply_rule(self, sy, nd, ca):
    ''' Balance supply/demand '''

    list_neg = self.sll | self.curt
#            list_dmnd_keys = self.dmnd.iterkeys()    
    prod = (# power output; negative if energy selling plant
                sum(self.pwr[sy, pp, ca]
                    * (-1 if pp in list_neg else 1)
                    for (pp, nd, ca)
                    in set_to_list(self.ppall_ndca, [None, nd, ca])))
    
    dmnd = 0
    exports = 0  
    if nd in nd_nt:
        prod += (
                # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                      / self.nd_weight[nd_2]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [None, nd, ca]))
                )
               
        dmnd += (self.dmnd[sy, nd, ca] 
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                  in set_to_list(self.st_ndca, [None, nd, ca]))) * (1 + self.grid_losses[nd, ca])
        
        exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                  / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, None, ca]))
    
    elif nd in nd_ch0:
        prod += (
                # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                      / self.nd_weight[nd_2]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [nd_nt, nd, ca]))
                + sum(get_transmission(sy, nd, nd_2, ca, False) * ((1 - self.grid_losses[nd, ca])**0.5)
                      / self.nd_weight[nd_2]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [nd_arch_el, nd, ca]))
                )
               
        dmnd += (self.dmnd[sy, nd, ca] 
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                  in set_to_list(self.st_ndca, [None, nd, ca]))) * (1 + self.grid_losses[nd, ca])
        
        exports += (sum(get_transmission(sy, nd, nd_2, ca, True) * ((1 - self.grid_losses[nd, ca])**0.5)
                  / self.nd_weight[nd]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [nd, nd_arch_el, ca]))
                + sum(get_transmission(sy, nd, nd_2, ca, True)
                  / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, nd_nt + nd_public_charging, ca])))
        
    elif nd in nd_arch_el:
        prod += (
                sum(get_transmission(sy, nd, nd_2, ca, False) * ((1 - self.grid_losses[nd, ca])**0.5)
                  / self.nd_weight[nd_2]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [None, nd, ca]))
                )
        dmnd += (self.dmnd[sy, nd, ca] 
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                  in set_to_list(self.st_ndca, [None, nd, ca])))
        
        exports += (sum(get_transmission(sy, nd, nd_2, ca, True) * ((1 - self.grid_losses[nd, ca])**0.5)
                    / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, nd_ch0, ca]))
                + sum(get_transmission(sy, nd, nd_2, ca, True)
                    / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, nd_arch_cl + nd_arch_ht + nd_arch_dsr + nd_arch_ev_res + nd_arch_ev_ind_oco, ca])))
    
    elif nd in nd_arch_cl:
          prod += (
                  # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                        / self.nd_weight[nd_2]
                        for (nd, nd_2, ca)
                        in set_to_list(self.ndcnn, [None, nd, ca]))
                )
                
          dmnd += (self.dmnd[sy, nd, ca] 
                  + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                    in set_to_list(self.st_ndca, [None, nd, ca])))
         
          exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                    / self.nd_weight[nd]
                    for (nd, nd_2, ca)
                    in set_to_list(self.ndcnn, [nd, None, ca]))
          
    elif nd in nd_arch_ht:
        prod += (
                # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                      / self.nd_weight[nd_2]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [None, nd, ca]))
                )
               
        dmnd += (self.dmnd[sy, nd, ca] 
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                  in set_to_list(self.st_ndca, [None, nd, ca])))
        
        exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                  / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, None, ca]))
    
    elif nd in nd_arch_dsr:
        prod += (
                # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                      / self.nd_weight[nd_2]
                      for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [None, nd, ca]))
                )
               
        dmnd += (self.dmnd[sy, nd, ca] 
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                  in set_to_list(self.st_ndca, [None, nd, ca])))
        
        exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                  / self.nd_weight[nd]
                  for (nd, nd_2, ca)
                  in set_to_list(self.ndcnn, [nd, None, ca]))
        
    elif nd in nd_arch_ev_res:
          prod += (
                  # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                        / self.nd_weight[nd_2]
                        for (nd, nd_2, ca)
                        in set_to_list(self.ndcnn, [None, nd, ca]))
                )
                
          dmnd += (self.dmnd[sy, nd, ca] 
                  + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                    in set_to_list(self.st_ndca, [None, nd, ca])))
         
          exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                    / self.nd_weight[nd]
                    for (nd, nd_2, ca)
                    in set_to_list(self.ndcnn, [nd, None, ca]))   
          
    elif nd in nd_arch_ev_ind_oco:
          prod += (
                  # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                        / self.nd_weight[nd_2]
                        for (nd, nd_2, ca)
                        in set_to_list(self.ndcnn, [None, nd, ca]))
                )
                
          dmnd += (self.dmnd[sy, nd, ca] 
                  + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                    in set_to_list(self.st_ndca, [None, nd, ca])))
         
          exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                    / self.nd_weight[nd]
                    for (nd, nd_2, ca)
                    in set_to_list(self.ndcnn, [nd, None, ca]))  


    # It is in national node connected!
    elif nd in nd_public_charging: 
          prod += (
                  # incoming inter-node transmission
                sum(get_transmission(sy, nd, nd_2, ca, False)
                        / self.nd_weight[nd_2]
                        for (nd, nd_2, ca)
                        in set_to_list(self.ndcnn, [None, nd, ca]))
                )
                
          dmnd += (self.dmnd[sy, nd, ca] 
                  + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                    in set_to_list(self.st_ndca, [None, nd, ca])))
         
          exports += sum(get_transmission(sy, nd, nd_2, ca, True)
                    / self.nd_weight[nd]
                    for (nd, nd_2, ca)
                    in set_to_list(self.ndcnn, [nd, None, ca]))  
          
        
    # demand of plants using ca as an input
    ca_cons = (po.ZeroConstant if not self.pp_ndcaca else
                sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                    for (pp, nd, ca_out, ca)
                    in set_to_list(self.pp_ndcaca,
                                  [list_pp_hp_eff + list_pp_dhw_bo + list_pp_hp_dhw_eff, nd, None, ca])))
                    # We update since we do direct convert of ca and cop from input data:                         
                            # [list_pp_hp + list_pp_dhw_bo + list_pp_hp_dhw, nd, None, ca])))
    
    ca_cons += (po.ZeroConstant if not self.pp_ndcaca else
                sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                    for (pp, nd, ca_out, ca)
                    in set_to_list(self.pp_ndcaca,
                                  [list_pp_cop, nd, None, ca])))
    ca_cons += (po.ZeroConstant if not self.pp_ndcaca else
                sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                    for (pp, nd, ca_out, ca)
                    in set_to_list(self.pp_ndcaca,
                                  [list_pp_dhw_cop, nd, None, ca])))  

    return prod == dmnd + ca_cons + exports            

self.cadd('supply', self.sy_ndca, rule=supply_rule)

pp_hp_aw = sorted(self.df_def_plant.loc[self.df_def_plant.pp.str.contains('HP_AW')].pp_id.tolist())
pp_hp_ww = sorted(self.df_def_plant.loc[self.df_def_plant.pp.str.contains('HP_WW')].pp_id.tolist())

nd_arch_ht = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('HT')].nd_id.unique().tolist())
nd_arch_cl = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('CL')].nd_id.unique().tolist())
nd_arch_dsr = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('DSR')].nd_id.unique().tolist())
nd_arch_ev = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('EV')].nd_id.unique().tolist())
nd_arch_res = sorted(self.df_def_node.loc[self.df_def_node.nd.str.contains('SFH|MFH')].nd_id.unique().tolist())
nd_arch_el_res = sorted(list(set(nd_arch_res) - set(nd_arch_cl) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev)))

# for sy in np.linspace(0.0,ml.m.sy.last(),num=int(ml.m.sy.last()+1.0)):
    # for nd_el, nd_ht, hp_aw in zip(nd_arch_el_res,nd_arch_ht,pp_hp_aw):
        # ml.m.trm[(int(sy), nd_el, nd_ht, 0)].setub(
                        # (ml.m.cap_pwr_leg[(hp_aw, 1)]/1)) # with aw and ww energy carier
        

# for sy in np.linspace(0.0,ml.m.sy.last(),num=int(ml.m.sy.last()+1.0)):
    # for nd_el, nd_ht, hp_aw in zip(nd_arch_el_res,nd_arch_ht,pp_hp_aw):
        # ml.m.trm[(int(sy), nd_el, nd_ht, 0)].setub(
                        # (ml.m.cap_pwr_leg[(hp_aw, 1)]/df_cop_35.loc[(df_cop_35.pp_id == hp_aw)].value.max())) # with aw and ww energy carier
###########
#%%

ml.m.init_solver()

# %

ml.io.init_output_tables()

ml.select_run(0)

# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

self = mlm
# %%

def set_ch_lol_capacity_fixed(ml, cap_mw=5000):
    """Set Swiss lost-load capacity to a fixed value in MW."""

    ch_lol_pp = ml.m.mps.dict_pp_id["CH_LOL_LIN"]

    ml.m.cap_pwr_leg[ch_lol_pp, 0] = float(cap_mw)


def run_model(run_id):
    # for irow in list(range(irow_0, len(ml.df_def_run))):
    #    run_id = irow

    ml.select_run(run_id)

    logger.info('reset_parameters')
    ml.m.reset_all_parameters()

    logger.info('select_swiss_scenarios')
    slct_ch = mlm.select_swiss_scenarios()

    logger.info('select_ev_scenarios')
    slct_ev = mlm.select_ev_scenarios()

    logger.info('select_v2g_scenarios')
    slct_v2g = mlm.select_v2g_scenarios()
    
    logger.info('select_hp_scenarios')
    slct_hp = mlm.select_hp_scenarios()
    
    logger.info('select_rf_scenarios')
    slct_rf = mlm.select_retrofit_scenarios()
    
    logger.info('select_cl_scenarios')
    slct_cl = mlm.select_cooling_scenarios()                 
    
    logger.info('select_demand_scenarios')
    slct_dpf = mlm.select_demand_profile_res()
    
    logger.info('select_tr_scenarios')
    slct_tr = mlm.select_transmission_scenarios()
 
    logger.info('select_ntc_scenarios')
    slct_ntc = mlm.select_transmission_scenarios_ntc()
    
    mlm.set_future_year(slct_ch=slct_ch, slct_ev=slct_ev, slct_v2g=slct_v2g, slct_tr=slct_tr, slct_ntc=slct_ntc, slct_hp=slct_hp, slct_rf=slct_rf, slct_cl=slct_cl, slct_dpf=slct_dpf)
  
    logger.info('keep_cap_new')
    mlm.keep_cap_new()
    #    #########################################
    ############### RUN MODEL ###############

    logger.info('fill_peaker_plants')
    
    ml.m.fill_peaker_plants(demand_factor=5,
                            #                            list_peak=[(ml.m.mps.dict_pp_id['CH_GAS_LIN'], 0)]
                            )
    
    logger.info('set_ch_lol_capacity_fixed')
    set_ch_lol_capacity_fixed(ml, cap_mw=5000)
    # logger.info('_limit_prof_to_cap')
    ml.m._limit_prof_to_cap()


    ml.perform_model_run(warmstart=False)
    # for fn in [f for f in list(os.walk('.'))[0][2] if f.startswith('tmp')]:
    #     os.remove(fn)
#
from grimsel.auxiliary.multiproc import run_parallel, run_sequential

print('RUNNING')
#run_sequential(ml, run_model)

# run_parallel(ml, run_model)
run_parallel(ml, run_model, 2, groupby=['swntc'], adjust_logger_levels=True)
#run_parallel(ml, run_model, 2, groupby=['swch','swntc'], adjust_logger_levels=True)
# run_parallel(ml, run_model, 3, groupby=['swv2g', 'swev','swtr','swntc'], adjust_logger_levels=True)

#%%
