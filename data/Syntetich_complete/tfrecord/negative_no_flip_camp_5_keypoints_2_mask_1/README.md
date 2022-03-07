
<h4> sets_config </h4>
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

<h4> dic_history </h4>

A small view of dic_history.pkl file of negative_no_flip_camp_5_keypoints_2_mask_1 configuration is shown below:

```
 {
 'train_0': {'pz_condition': 'pz101', 'img_condition': '00000_8bit.png', 'pz_target': 'pz103', 'img_target': '00000_8bit.png', 'id_in_tfrecord': 'train_0'}
 'train_1': {'pz_condition': 'pz101', 'img_condition': '00005_8bit.png', 'pz_target': 'pz103', 'img_target': '00005_8bit.png', 'id_in_tfrecord': 'train_1'}
 .
 .
 .
 'valid_0': {'pz_condition': 'pz102', 'img_condition': '00000_8bit.png', 'pz_target': 'pz111', 'img_target': '00000_8bit.png', 'id_in_tfrecord': 'valid_0'}
 'vaid_1': {'pz_condition': 'pz102', 'img_condition': '00005_8bit.png', 'pz_target': 'pz111', 'img_target': '00005_8bit.png', 'id_in_tfrecord': 'valid_1'}
 .
 .
 .
 'test_0': {'pz_condition': 'pz104', 'img_condition': '00000_8bit.png', 'pz_target': 'pz108', 'img_target': '00000_8bit.png', 'id_in_tfrecord': 'test_0'}
 'test_1': {'pz_condition': 'pz104', 'img_condition': '00005_8bit.png', 'pz_target': 'pz108', 'img_target': '00005_8bit.png', 'id_in_tfrecord': 'test_1'}
 .
 .
 .
}
```




