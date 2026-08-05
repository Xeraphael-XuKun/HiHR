
############ AGReID
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/fastreid39/bin/python tools/train_net.py --config-file ./configs/AGReID/HiHR.yml MODEL.DEVICE 'cuda:0'

############ AGReID.v2
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/fastreid39/bin/python tools/train_net.py --config-file ./configs/AGReIDv2/HiHR.yml MODEL.DEVICE 'cuda:0'
############

############ CAGRO
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/fastreid39/bin/python tools/train_net.py --config-file ./configs/CARGO/HiHR.yml MODEL.DEVICE 'cuda:0'
############

############ G2APS_ReID
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/fastreid39/bin/python tools/train_net.py --config-file ./configs/G2APS_ReID/HiHR.yml MODEL.DEVICE 'cuda:0'
############

############ LAGPeR
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/fastreid39/bin/python tools/train_net.py --config-file ./configs/LAGPeR/HiHR.yml MODEL.DEVICE 'cuda:0'
############
