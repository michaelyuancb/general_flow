
import numpy as np
import torch
import pdb
from .point_transformer_gpu import DataTransforms
from scipy.linalg import expm, norm


@DataTransforms.register_module()
class RandomRotate_KPST(object):
    def __init__(self, angle=[0, 0, 1], **kwargs):
        self.angle = angle

    def __call__(self, data):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        data['pos'] = np.dot(data['pos'], np.transpose(R))       # (N, 3)
        data['dtraj'] = np.dot(data['dtraj'], np.transpose(R))   # (Q, T=5, 3)

        return data


@DataTransforms.register_module()
class RandomScale_KPST(object):
    def __init__(self, scale=[0.8, 1.2],
                 scale_anisotropic=False,
                 scale_xyz=[True, True, True],
                 mirror=[-1, -1, -1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.anisotropic = scale_anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        if len(scale) == 1:
            scale = scale.repeat(3)
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        data['pos'] *= scale
        data['dtraj'] *= scale
        return data

    def __repr__(self):
        return 'RandomScale_KPST(scale_low: {}, scale_high: {})'.format(self.scale_min, self.scale_max)


@DataTransforms.register_module()
class CoordNorm_KPST(object):
    def __init__(self, **kwargs):
        self.shape_np = np.zeros((1, 1, 3))

    def __call__(self, data):
        norm_coord = data['pos'].mean(0)          
        data['pos'] = data['pos'] - norm_coord    
        data['dtraj'] = data['dtraj'] - (norm_coord + self.shape_np)
        return data

    def __repr__(self):
        return 'CoordNorm_KPST, pos.mean=(0, 0, 0)'
    

@DataTransforms.register_module()
class KPSCoordNorm_KPST(object):
    def __init__(self, **kwargs):
        self.shape_np = np.zeros((1, 1, 3))

    def __call__(self, data):
        norm_coord = data['dtraj'][:, 0, :].mean(0)   # (Q, T, 3)
        data['pos'] = data['pos'] - norm_coord    
        data['dtraj'] = data['dtraj'] - (norm_coord + self.shape_np)
        return data

    def __repr__(self):
        return 'CoordNorm_KPST, pos.mean=(0, 0, 0)'


@DataTransforms.register_module()
class RandomScaleAndTranslate_KPST(object):
    def __init__(self,
                 scale=[0.9, 1.1],
                 shift=[0., 0., 0.],
                 scale_xyz=[1, 1, 1],
                 scale_anisotropic=False,
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.shift_up = shift
        self.shift_dn = [-x for x in shift]
        self.anisotropic = scale_anisotropic

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        if len(scale) == 1:
            scale = scale.repeat(3)
        scale *= self.scale_xyz

        shift = np.random.uniform(self.shift_dn, self.shift_up, 3)
        data['pos'] = data['pos'] * scale + shift 
        data['dtraj'] = data['dtraj'] * scale + shift

        return data

    def __repr__(self):
        return 'RandomScaleAndTranslate_KPST(scale={}, shift={})'.format(self.scale, self.shift)


@DataTransforms.register_module()
class RandomMask_KPST(object):
    # Random Mask a Unit Ball
    def __init__(self,
                 mask_unit_length=0.25,
                 mask_prob=0.8,
                 **kwargs):
        self.mask_unit_length = mask_unit_length
        self.mask_prob = mask_prob

    def __call__(self, data):

        p = np.random.uniform(0, 1)
        if p > self.mask_prob:
            return data

        # pdb.set_trace()
        pos, feat = data['pos'], data['x']                            # (N, 3)
        n_points = pos.shape[0]
        limit_points = data['dtraj'][:, 0, :]        # (Q, 3)
        half_length = self.mask_unit_length / 2
        attempts = 0

        while attempts < 3:
            center = pos[np.random.choice(pos.shape[0])]
            dis_limit = np.linalg.norm(limit_points - center, axis=1)
            if np.sum(dis_limit < half_length) > 0:
                attempts += 1
                continue 
            mask = np.linalg.norm(pos - center, axis=1) < half_length
            pos, feat = pos[~mask], feat[~mask]
            if pos.shape[0] < n_points:
                num_to_add = n_points - pos.shape[0]
                indices_to_add = np.random.choice(pos.shape[0], num_to_add, replace=True)
                pos_to_add, feat_to_add = pos[indices_to_add], feat[indices_to_add]
                pos = np.concatenate((pos, pos_to_add), axis=0)
                feat = np.concatenate((feat, feat_to_add), axis=0)
            break
            
        data['pos'] = pos
        data['x'] = feat
        return data

    def __repr__(self):
        return 'RandomMask_KPST(mask_unit_length={})'.format(self.mask_unit_length)