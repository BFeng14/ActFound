# all data necessary for plot is in "result_plot" dir, please replace all "/home/fengbin/meta_delta" in the code into "./result_plot"

## fig2: in-domain
#python metaddg_figure_indomain.py r2 & # 2.a chembl，bindingdb in-domain, correlation, 16-shot
#python metaddg_figure_indomain.py rmse & # 2.b chembl，bindingdb in-domain, RMSE, 16-shot
#
#python metaddg_figure_indomain_fsmol.py r2 # 2.c FSMOL in-domain, correlation, 16,32,64,128-shot
#python metaddg_figure_indomain_fsmol.py rmse # supplement FSMOL in-domain, RMSE, 16,32,64,128-shot
#python metaddg_figure_indomain_fsmol.py R2os # supplement FSMOL in-domain, R2os, 16,32,64,128-shot
#
#python metaddg_figure_indomain_pqsar.py r2 # 2.c FSMOL in-domain, correlation, 16,32,64,128-shot
#python metaddg_figure_indomain_pqsar.py rmse # supplement FSMOL in-domain, RMSE, 16,32,64,128-shot
#
#python metaddg_figure_activity_domain.py r2 16 # 2.e pretrain on chembl. Test on ChEMBL-Activity, Correlation, 16-shot
#python metaddg_figure_activity_domain.py rmse 16 # 2.f pretrain on chembl. Test on ChEMBL-Activity, RMSE, 16-shot
#
##python t-sne_plot.py metadelta # 2.g Ligand t-sne, (g1) with delta learning(meta-delta);
##python t-sne_plot.py maml # 2.g Ligand t-sne, (g2) without delta(MAML)
#
#
## fig3: cross-domain
#python metaddg_figure_cross_domain.py r2 & # 3.a chembl to bindingdb, bindingdb to chembl correlation, 16-shot
#python metaddg_figure_cross_domain.py rmse & # 3.b chembl to bindingdb, bindingdb to chembl RMSE, 16-shot
#
#python metaddg_figure_cross_domain_kibadavis.py r2 KIBA & # 3.c chembl to KIBA, bindingdb to KIBA, correlation, 16-shot
#python metaddg_figure_cross_domain_kibadavis.py rmse KIBA & # 3.d chembl to KIBA, bindingdb to KIBA, RMSE, 16-shot
#
#python metaddg_figure_cross_domain_kibadavis.py r2 Davis & # 3.e chembl to Davis, bindingdb to Davis, correlation, 16-shot
#python metaddg_figure_cross_domain_kibadavis.py rmse Davis & # 3.f chembl to Davis, bindingdb to Davis, RMSE, 16-shot
#
#python coor_plot.py # 3.g zero-step loss & r2, scatter plot

# fig4: expert-domain
# Attention: all figure in expert-domain was previously ploted by EXCEL, need to write new python script

python metaddg_figure_fep_domain.py r2 fep chembl &
python metaddg_figure_fep_domain.py rmse fep chembl &
python metaddg_figure_fep_domain.py r2 fep_opls4 chembl &
python metaddg_figure_fep_domain.py rmse fep_opls4 chembl &
python metaddg_figure_fep_domain.py r2 fep bdb &
python metaddg_figure_fep_domain.py rmse fep bdb &
python metaddg_figure_fep_domain.py r2 fep_opls4 bdb &
python metaddg_figure_fep_domain.py rmse fep_opls4 bdb &