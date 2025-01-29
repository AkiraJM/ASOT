python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method EMD
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method Sinkhorn
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT --epochs 2000
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT_ML --epochs 2000 --ml 100.0
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT_k
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method sliced_Wass

python exp_draw_curves.py --dataset MUTAG --epochs 2000 --ml 100.0
pause