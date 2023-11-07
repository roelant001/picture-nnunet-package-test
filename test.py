import os
import glob
from picture_nnunet_package.doInference import do_segmentation
from nnunet.evaluation.evaluator import aggregate_scores

sesDirs = glob.glob(f'./test_data/LUMIERE/Patient-025/*')
for sesDir in sesDirs:
    sesID = os.path.split(sesDir)[1]
    imgs = {}
    for mod in ['CT1','T1','T2','FLAIR']:
        imgs[mod] = glob.glob(sesDir+'/'+mod+'*.nii.gz')[0]
    if 'week-000' in sesID:
        sessionType = 'preop'
    else:
        sessionType = 'postop_beta'

    do_segmentation(imgs['CT1'], t1=imgs['T1'], t2=imgs['T2'], flair=imgs['FLAIR'], sessionType =sessionType, remove_intermediate_files = False, mni=False, wdir_postfix='random',skip_skullstrip=True)
    
    my_seg = glob.glob(f'{sesDir}/segmentation_native*.nii.gz')[0]
    ref_seg = glob.glob(f'{sesDir}/reference_segmentation_native*.nii.gz')[0]
    scores = aggregate_scores([(my_seg,ref_seg)],json_output_file="./summary.json",labels=(0,1,2,3))
    print(scores['mean'])
    for label in scores['mean'].keys():
        if scores['mean'][label]['Total Positives Reference']>5:
            assert  scores['mean'][label]['Dice'] >0.8
