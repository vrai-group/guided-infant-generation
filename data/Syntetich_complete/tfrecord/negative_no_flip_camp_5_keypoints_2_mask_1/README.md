The sets_config.pkl file of negative_no_flip_camp_5_keypoints_2_mask_1 configuration is shown below:

```
 {
'general': {'campionamento': 5, 
 	    'radius_keypoints_pose (r_k)': 2, 
	    'radius_keypoints_mask': 1, 
	    'radius_head_mask (r_h)': 40, 
	    'dilatation': 35, 'flip': False, 
	    'pairing_mode': 'negative'},
	      
'train': {'name_file': 
	  'Syntetich_train.tfrecord', 
	  'list_pz': [101, 103, 105, 106, 107, 109, 110, 112], 
	  'tot': 11200}, 
	   
'valid': {'name_file': 'Syntetich_valid.tfrecord', 
	 'list_pz': [102, 111], 
	 'tot': 400},
 
 'test': {'name_file': 'Syntetich_test.tfrecord',
          'list_pz': [104, 108], 
	  'tot': 400}
}
```
