def predict_conditional_GAN (config):
    babypose_obj = BabyPose()

    # Preprocess Dataset
    dataset = babypose_obj.get_unprocess_dataset(config.name_tfrecord_train)
    dataset = babypose_obj.get_preprocess_GAN_dataset(dataset)
    dataset = dataset.batch(1)
    dataset = iter(dataset)

    dataset_valid = babypose_obj.get_unprocess_dataset(config.name_tfrecord_valid)
    dataset_valid = babypose_obj.get_preprocess_GAN_dataset(dataset_valid)
    dataset_valid = dataset_valid.batch(1)
    dataset_valid = iter(dataset_valid)

    dataset_test = babypose_obj.get_unprocess_dataset(config.name_tfrecord_test)
    dataset_test = babypose_obj.get_preprocess_GAN_dataset(dataset_test)
    dataset_test = dataset_test.batch(1)
    dataset_test = iter(dataset_test)

    # Carico il modello preaddestrato G1
    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_path,'Model_G1_epoch_030-loss_0.000312-ssim_0.788846-mask_ssim_0.982531-val_loss_0.000795-val_ssim_0.730199_val_mask_ssim_0.946310.hdf5'))
    # Carico il modello preaddestrato GAN
    # G2
    model_G2 = G2.build_model()  # architettura Generatore G2
    model_G2.load_weights(os.path.join(config.weigths_path, 'a.hdf5'))
    # D
    model_D = Discriminator.build_model()
    # model_D.load_weights(os.path.join(config.weigths_path, 'b.hdf5'))


    # cnt2 = 0
    # cnt = 0
    # p = []  # per raccogliere le pose del train
    # raw1 = []  # per raccogliere le target del train
    #
    # for id_batch in range(int(config.dataset_train_len / 1)):
    #
    #     batch = next(dataset_valid)
    #     image_raw_0 = batch[0]  # [batch, 96, 128, 1]
    #     image_raw_1 = batch[1]  # [batch, 96,128, 1]
    #     pose_1 = batch[2]  # [batch, 96,128, 14]
    #     mask_1 = batch[3]  # [batch, 96,128, 1]
    #     mask_0 = batch[4]  # [batch, 96,128, 1]
    #     pz_0 = batch[5]  # [batch, 1]
    #     pz_1 = batch[6]  # [batch, 1]
    #     name_0 = batch[7]  # [batch, 1]
    #     name_1 = batch[8]  # [batch, 1]
    #
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     print(pz_0, '-', pz_1)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz3" and pz_1 == "pz34": #salviamo la posa del pz_1
    #
    #             p.append(pose_1)
    #
    #         if len(p) >= 5:
    #             print("Terminata raccolta pose")
    #             for id_batch in range(int(config.dataset_train_len / 1)):
    #
    #                 batch = next(dataset_valid)
    #                 image_raw_0 = batch[0]  # [batch, 96, 128, 1]
    #                 image_raw_1 = batch[1]  # [batch, 96,128, 1]
    #                 pose_1 = batch[2]  # [batch, 96,128, 14]
    #                 mask_1 = batch[3]  # [batch, 96,128, 1]
    #                 mask_0 = batch[4]  # [batch, 96,128, 1]
    #                 pz_0 = batch[5]  # [batch, 1]
    #                 pz_1 = batch[6]  # [batch, 1]
    #                 name_0 = batch[7]  # [batch, 1]
    #                 name_1 = batch[8]  # [batch, 1]
    #
    #
    #                 if pz_0 == "pz110":
    #                     # G1
    #                     input_G1 = tf.concat([image_raw_0, p[cnt2]], axis=-1)  # [batch, 96, 128, 15]
    #                     output_G1 = model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]
    #                     output_G1 = tf.cast(output_G1, dtype=tf.float16)
    #
    #                     # G2
    #                     input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
    #                     output_G2 = model_G2(input_G2)  # [batch, 96, 128, 1]
    #                     output_G2 = tf.cast(output_G2, dtype=tf.float16)
    #                     refined_result = output_G1 + output_G2
    #
    #                     # Unprocess
    #                     image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 350, 32765.5)
    #                     image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()
    #
    #                     image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 350, 32765.5)
    #                     image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()
    #
    #                     pose_1 = p[cnt2].numpy()[0]
    #                     pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    #                     pose_1 = pose_1 / 2
    #                     pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
    #                     pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    #                     pose_1 = tf.cast(pose_1, dtype=tf.float32)
    #
    #                     mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)
    #                     mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255
    #
    #                     refined_result = \
    #                     tf.cast(utils_wgan.unprocess_image(refined_result, 900, 32765.5), dtype=tf.uint16)[0]
    #
    #                     result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.math.reduce_max(refined_result))
    #                     print(result)
    #
    #                     output_G1 = tf.clip_by_value(utils_wgan.unprocess_image(output_G1, 350, 32765.5),
    #                                                  clip_value_min=0,
    #                                                  clip_value_max=32765)
    #                     output_G1 = tf.cast(output_G1, dtype=tf.uint16)[0]
    #
    #                     output_G2 = tf.clip_by_value(utils_wgan.unprocess_image(output_G2, 350, 32765.5),
    #                                                  clip_value_min=0,
    #                                                  clip_value_max=32765)
    #                     output_G2 = tf.cast(output_G2, dtype=tf.uint16)[0]
    #
    #                     # Save img
    #                     # import cv2
    #                     # refined_result = tf.cast((refined_result*32765.5)+350, dtype=tf.uint16)[0]
    #                     # cv2.imwrite("t.png", refined_result.numpy())
    #
    #                     # Predizione D
    #                     # input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
    #                     #                     axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
    #                     # output_D = self.model_D(input_D)  # [batch * 3, 1]
    #                     # output_D = tf.reshape(output_D, [-1])  # [batch*3]
    #                     # output_D = tf.cast(output_D, dtype=tf.float16)
    #                     # D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]
    #
    #                     fig = plt.figure(figsize=(10, 10))
    #                     columns = 6
    #                     rows = 1
    #                     imgs = [output_G1, output_G2, refined_result, pose_1, image_raw_1, image_raw_0]
    #                     for i in range(1, columns * rows + 1):
    #                         fig.add_subplot(rows, columns, i)
    #                         plt.imshow(imgs[i - 1])
    #                     plt.show()
    #                     cnt2+=1


    for id_batch in range(int(config.dataset_train_len / 1)):

        batch = next(dataset)
        if id_batch > 100:
            image_raw_0 = batch[0]  # [batch, 96, 128, 1]
            image_raw_1 = batch[1]  # [batch, 96,128, 1]
            pose_1 = batch[2]  # [batch, 96,128, 14]
            mask_1 = batch[3]  # [batch, 96,128, 1]
            mask_0 = batch[4]  # [batch, 96,128, 1]
            pz_0 = batch[5]  # [batch, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_0 = batch[7]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))


            print(name_1)
            print(name_0)


            # G1
            input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)  # [batch, 96, 128, 15]
            output_G1 = model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]
            output_G1 = tf.cast(output_G1, dtype=tf.float16)

            # G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2 = model_G2(input_G2)  # [batch, 96, 128, 1]
            output_G2 = tf.cast(output_G2, dtype=tf.float16)
            refined_result = output_G1 + output_G2

            # Predizione D
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            print("Reale? ", D_neg_refined_result)


            # Unprocess
            image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5)
            image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()

            image_raw_1 = utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5)
            image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()

            pose_1 = pose_1.numpy()[0]
            pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
            pose_1 = pose_1 / 2
            pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
            pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
            pose_1 = tf.cast(pose_1, dtype=tf.float32)

            mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)
            mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255

            refined_result = utils_wgan.unprocess_image(refined_result, mean_0, 32765.5)
            refined_result= tf.cast(refined_result, dtype=tf.uint16)[0]

            # result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.math.reduce_max(refined_result))
            # print(result)


            output_G2 = tf.clip_by_value(utils_wgan.unprocess_image(output_G2, mean_0, 32765.5), clip_value_min=0,
                                         clip_value_max=32765)
            output_G2 = tf.cast(output_G2, dtype=tf.uint8)[0]

            output_G1 = tf.clip_by_value(utils_wgan.unprocess_image(output_G1, mean_1, 32765.5), clip_value_min=0,
                                         clip_value_max=32765)
            output_G1 = tf.cast(output_G1, dtype=tf.uint16)[0]



            #Save img
            # import cv2
            # refined_result = tf.cast(refined_result, dtype=tf.uint8)
            # cv2.imwrite("t.png", refined_result.numpy())


            # grid.save_image(tf.reshape(refined_result,(2,96,128)),
            #                 "./t.png")

            fig = plt.figure(figsize=(10, 10))
            columns = 6
            rows = 1
            imgs = [output_G1, output_G2, refined_result, pose_1, image_raw_0, image_raw_1]
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(imgs[i - 1], cmap='gray')
            plt.show()