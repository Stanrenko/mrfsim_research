import unittest
from image_series import *
from utils_mrf import *



try:
    import cupy as cp
    gpu_test=True
except:
    gpu_test = False

KDATA_FILE="kdata_squareph_testcase.npy"
VOLUMES_FILE="volumes_squareph_testcase.npy"
GROUNDTRUTH_FILE="groundtruthMaps_squareph_testcase.pkl"
#KDATA_FILE="./tests/kdata_squareph_testcase.npy"
KDATA_FILE_1COIL="1coil_squareph_testcase_kdata.npy"
#KDATA_FILE_MULTI="./tests/kdata_multi_squareph_testcase.npy"
KDATA_FILE_4COIL="4coil_squareph_testcase_kdata.npy"
B1_FILE_1COIL = "1coil_squareph_testcase_b1.npy"
B1_FILE_4COIL = "4coil_squareph_testcase_b1.npy"
VOL_FILE_1COIL = "1coil_squareph_testcase_volumes.npy"
VOL_FILE_4COIL = "4coil_squareph_testcase_volumes.npy"


# ntimesteps = 175
# nb_allspokes = 1400
# npoint = 256
# nb_slices = 8
# undersampling_factor=1
#
# sampling_mode="stack"
# data = np.load(KDATA_FILE_MULTI)
# kdata_all_channels_all_slices = data.reshape(-1, npoint)
# density = np.abs(np.linspace(-1, 1, npoint))
# kdata_all_channels_all_slices = (kdata_all_channels_all_slices*density).reshape(4,1400,-1,npoint)
# #kdata_all_channels_all_slices=np.expand_dims(kdata_all_channels_all_slices,axis=0)
# np.save(KDATA_FILE_4COIL,kdata_all_channels_all_slices)


#
# volumes=np.load("./tests/"+VOL_FILE_4COIL)
#
# animate_images(np.abs(volumes[:,2,:,:]))

class MyTestCase(unittest.TestCase):
    def test_coil_sensi_unichannel(self):

        kdata_all_channels_all_slices = np.load(KDATA_FILE_1COIL)
        data_shape = kdata_all_channels_all_slices.shape
        undersampling_factor = 1
        nb_channels = data_shape[0]
        nb_allspokes = data_shape[1]
        npoint = data_shape[-1]
        nb_slices = data_shape[2]
        image_size = (4, 64, 64)
        incoherent=False
        mode = None


        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        res = 8
        b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                     useGPU=False, light_memory_usage=True)

        np.save(B1_FILE_1COIL, b1_all_slices)
        self.assertEqual(b1_all_slices.shape[0],nb_channels)
        self.assertEqual(b1_all_slices.shape[1], image_size[0])
        self.assertEqual(b1_all_slices.shape[2], image_size[1])
        self.assertEqual(b1_all_slices.shape[3], image_size[2])
        self.assertTrue(np.linalg.norm(np.squeeze(np.abs(b1_all_slices))-np.ones(image_size))<0.0000001)

        #Light memory usage
        b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                     useGPU=False, light_memory_usage=False)


        self.assertEqual(b1_all_slices.shape[0], nb_channels)
        self.assertEqual(b1_all_slices.shape[1], image_size[0])
        self.assertEqual(b1_all_slices.shape[2], image_size[1])
        self.assertEqual(b1_all_slices.shape[3], image_size[2])
        self.assertTrue(np.linalg.norm(np.squeeze(np.abs(b1_all_slices)) - np.ones(image_size)) < 0.0000001)

        if gpu_test:
            b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                         useGPU=True, light_memory_usage=True)

            self.assertEqual(b1_all_slices.shape[0], nb_channels)
            self.assertEqual(b1_all_slices.shape[1], image_size[0])
            self.assertEqual(b1_all_slices.shape[2], image_size[1])
            self.assertEqual(b1_all_slices.shape[3], image_size[2])
            self.assertTrue(np.linalg.norm(np.squeeze(np.abs(b1_all_slices)) - np.ones(image_size)) < 0.0000001)

            # Light memory usage
            b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                         useGPU=True, light_memory_usage=False)

            self.assertEqual(b1_all_slices.shape[0], nb_channels)
            self.assertEqual(b1_all_slices.shape[1], image_size[0])
            self.assertEqual(b1_all_slices.shape[2], image_size[1])
            self.assertEqual(b1_all_slices.shape[3], image_size[2])
            self.assertTrue(np.linalg.norm(np.squeeze(np.abs(b1_all_slices)) - np.ones(image_size)) < 0.0000001)

    def test_coil_sensi_multichannel(self):

        kdata_all_channels_all_slices = np.load(KDATA_FILE_4COIL)
        data_shape = kdata_all_channels_all_slices.shape
        undersampling_factor = 1
        nb_channels = data_shape[0]
        nb_allspokes = data_shape[1]
        npoint = data_shape[-1]
        nb_slices = data_shape[2]
        image_size = (4, 64, 64)
        incoherent=False
        mode = None


        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        res = 8
        b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                     useGPU=False, light_memory_usage=True)
        b1_ref=np.load(B1_FILE_4COIL)
        self.assertEqual(b1_all_slices.shape[0],nb_channels)
        self.assertEqual(b1_all_slices.shape[1], image_size[0])
        self.assertEqual(b1_all_slices.shape[2], image_size[1])
        self.assertEqual(b1_all_slices.shape[3], image_size[2])

        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, :32, :32]),0.4058847,6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 3, :32, :32]), 0.4058847, 6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, 32:, 32:]), 0.8341962, 6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[3, 1, :32, :32]), 0.861597, 6)
        self.assertTrue(np.linalg.norm(b1_all_slices-b1_ref) < 0.0000001)

        #Light memory usage
        b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                     useGPU=False, light_memory_usage=False)

        self.assertEqual(b1_all_slices.shape[0], nb_channels)
        self.assertEqual(b1_all_slices.shape[1], image_size[0])
        self.assertEqual(b1_all_slices.shape[2], image_size[1])
        self.assertEqual(b1_all_slices.shape[3], image_size[2])

        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, :32, :32]), 0.4058847, 6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 3, :32, :32]), 0.4058847, 6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, 32:, 32:]), 0.8341962, 6)
        self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[3, 1, :32, :32]), 0.861597, 6)
        self.assertTrue(np.linalg.norm(b1_all_slices - b1_ref) < 0.0000001)

        if gpu_test:
            b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                         useGPU=True, light_memory_usage=True)

            self.assertEqual(b1_all_slices.shape[0], nb_channels)
            self.assertEqual(b1_all_slices.shape[1], image_size[0])
            self.assertEqual(b1_all_slices.shape[2], image_size[1])
            self.assertEqual(b1_all_slices.shape[3], image_size[2])

            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, :32, :32]), 0.4058847, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 3, :32, :32]), 0.4058847, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, 32:, 32:]), 0.8341962, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[3, 1, :32, :32]), 0.861597, 6)
            self.assertTrue(np.linalg.norm(b1_all_slices - b1_ref) < 0.0000001)

            # Light memory usage
            b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                         useGPU=True, light_memory_usage=False)

            self.assertEqual(b1_all_slices.shape[0], nb_channels)
            self.assertEqual(b1_all_slices.shape[1], image_size[0])
            self.assertEqual(b1_all_slices.shape[2], image_size[1])
            self.assertEqual(b1_all_slices.shape[3], image_size[2])

            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, :32, :32]), 0.4058847, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 3, :32, :32]), 0.4058847, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[0, 1, 32:, 32:]), 0.8341962, 6)
            self.assertAlmostEqual(np.mean(np.abs(b1_all_slices)[3, 1, :32, :32]), 0.861597, 6)
            self.assertTrue(np.linalg.norm(b1_all_slices - b1_ref) < 0.0000001)

    # def test_volumes_unichannel(self):
    #     kdata_all_channels_all_slices = np.load(KDATA_FILE_1COIL)
    #     b1_all_slices = np.load(B1_FILE_1COIL)
    #     data_shape = kdata_all_channels_all_slices.shape
    #     ntimesteps = 175
    #     undersampling_factor = 1
    #     nb_channels = data_shape[0]
    #     nb_allspokes = data_shape[1]
    #     npoint = data_shape[-1]
    #     nb_slices = data_shape[2]
    #     image_size = (4, 64, 64)
    #     incoherent = False
    #     mode = None
    #
    #     radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
    #                            nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    #
    #     volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj,
    #                                                             image_size,
    #                                                             b1=b1_all_slices, density_adj=False,
    #                                                             ntimesteps=ntimesteps,
    #                                                             useGPU=False, normalize_kdata=True,
    #                                                             memmap_file=None,
    #                                                             light_memory_usage=False,
    #                                                             normalize_volumes=True)
    #     np.save(VOL_FILE_1COIL, volumes_all)

    def test_volumes_multichannel(self):
        kdata_all_channels_all_slices = np.load(KDATA_FILE_4COIL)
        b1_all_slices = np.load(B1_FILE_4COIL)
        data_shape = kdata_all_channels_all_slices.shape
        ntimesteps=175
        undersampling_factor = 1
        nb_channels = data_shape[0]
        nb_allspokes = data_shape[1]
        npoint = data_shape[-1]
        nb_slices = data_shape[2]
        image_size = (4, 64, 64)
        incoherent = False
        mode = None

        volumes_ref = np.load(VOL_FILE_4COIL)

        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj,
                                                                image_size,
                                                                b1=b1_all_slices, density_adj=False,
                                                                ntimesteps=ntimesteps,
                                                                useGPU=False, normalize_kdata=True,
                                                                memmap_file=None,
                                                                light_memory_usage=False,
                                                                normalize_volumes=True)
        print(np.linalg.norm(volumes_all - volumes_ref))
        self.assertTrue(np.linalg.norm(volumes_all - volumes_ref) < 0.000001)

        volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj,
                                                                image_size,
                                                                b1=b1_all_slices, density_adj=False,
                                                                ntimesteps=ntimesteps,
                                                                useGPU=False, normalize_kdata=True,
                                                                memmap_file=None,
                                                                light_memory_usage=True,
                                                                normalize_volumes=True)
        print(np.linalg.norm(volumes_all - volumes_ref))
        self.assertTrue(np.linalg.norm(volumes_all - volumes_ref) < 0.000001)

        if gpu_test:
            volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj,
                                                                    image_size,
                                                                    b1=b1_all_slices, density_adj=False,
                                                                    ntimesteps=ntimesteps,
                                                                    useGPU=True, normalize_kdata=True,
                                                                    memmap_file=None,
                                                                    light_memory_usage=False,
                                                                    normalize_volumes=True)

            self.assertTrue(np.linalg.norm(volumes_all - volumes_ref) < 0.0000001)

            volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj,
                                                                    image_size,
                                                                    b1=b1_all_slices, density_adj=False,
                                                                    ntimesteps=ntimesteps,
                                                                    useGPU=True, normalize_kdata=True,
                                                                    memmap_file=None,
                                                                    light_memory_usage=True,
                                                                    normalize_volumes=True)

            self.assertTrue(np.linalg.norm(volumes_all - volumes_ref) < 0.0000001)



if __name__ == '__main__':
    unittest.main()
