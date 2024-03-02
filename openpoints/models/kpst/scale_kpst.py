import copy
import pdb
from typing import List
import torch
import torch.nn as nn
from ..build import MODELS, build_model_from_cfg
from openpoints.models.layers.activation import create_act


class BaseKPST(nn.Module):
    def __init__(self,
                 loss_args=None,
                 clip_align_args=None,
                 encoder_args=None,
                 decoder_args=None,
                 cvae_args=None,
                 **kwargs):
        super().__init__()

        if clip_align_args.get('fusion', None) is None:
            raise ValueError("clip_align_args must has 'fusion' attribute.")
        if clip_align_args.fusion not in ['late', 'early', 'both']:
            raise ValueError(f"clip_align_args.fusion must in ['late', 'early', 'both'], but get '{clip_align_args.fusion}'")
        self.clip_fusion = clip_align_args.fusion

        if self.clip_fusion in ['early', 'both']: 
            encoder_args.in_channels = encoder_args.in_channels + clip_align_args.aligned_width

        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        else:
            self.decoder = None

        # RGBXYZ for PointNeXt
        self.query_embedding = torch.nn.Parameter(torch.randn(1, 1, 6), requires_grad=True)
        clip_dim = [512] + clip_align_args.hidden_dim + [clip_align_args.aligned_width]
        clip_module = nn.Sequential()
        for i in range(len(clip_dim)-1):
            clip_module.add_module(f'clip_align_linear_{i}', nn.Linear(clip_dim[i], clip_dim[i+1]))
            if i == len(clip_dim)-2: continue
            clip_module.add_module(f'clip_align_act_{i}', create_act(clip_align_args.act))
        if clip_align_args.norm is not None:
            clip_module.add_module('clip_align_bn', nn.BatchNorm1d(clip_dim[-1]))
        if clip_align_args.dropout > 0:
            clip_module.add_module('clip_align_dropout', nn.Dropout(clip_align_args.dropout))
        self.clip_alignment = clip_module

        if hasattr(self.decoder, 'out_channels'):
            in_channels = self.decoder.out_channels
        elif hasattr(self.encoder, 'out_channels'):
            in_channels = self.encoder.out_channels
        else:
            in_channels = cvae_args.get('in_channels', None)

        if cvae_args is not None:
            if clip_align_args.fusion == 'late':
                cvae_args.condition_dim = in_channels + clip_align_args.aligned_width + 3
            elif clip_align_args.fusion == 'both':
                cvae_args.condition_dim = in_channels + clip_align_args.aligned_width_1 + 3
            else:
                cvae_args.condition_dim = in_channels + 3
            self.cvae = build_model_from_cfg(cvae_args)
        

    def get_context(self, pos, feat, query, n_b, n_q, text_emb=None, return_all=False):
        query_pos = torch.concat([pos, query], axis=-2)                                       # (B, N+Q, 3)
        query_feat = torch.concat([feat, self.query_embedding.repeat(n_b, n_q, 1)], axis=-2)  # (B, N+Q, 6)
        if self.clip_fusion in ['early', 'both']:
            query_feat = torch.concat([query_feat, text_emb.unsqueeze(-2).repeat(1, query_feat.shape[1], 1)], axis=-1)          # (B, N+Q, 6+F_aligned_dim)

        p, f = self.encoder({'pos': query_pos, 'x': query_feat.transpose(1, 2)})
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)   # (B, F, N+Q)

        if return_all is True:
            return f.transpose(1, 2)                 # (B, N+Q, F)
        else: 
            return f[:, :, -n_q:].transpose(1, 2)    # (B, Q, F)


    def forward(self, pos, feat, text_feat, dtraj, pack=None):
        pass


    def inference(self, pos, feat, text_feat, query, num_sample=10):
        if self.training:
            raise RuntimeError("model needs to be in eval mode for inference")
        
        # query             # (B, Q, 3)
        n_b = query.shape[0]
        n_q = query.shape[1]     
        query = query.reshape(n_b, n_q, -1)                         

        ft_text = self.clip_alignment(text_feat)            
        ft_query = self.get_context(pos, feat, query, n_b, n_q, text_emb=ft_text)  
        if self.clip_fusion == 'late':
            ft_context = torch.cat([query, ft_text.unsqueeze(1).repeat(1, n_q, 1), ft_query], dim=-1) 
        elif self.clip_fusion == 'both':
            ft_query_global = self.clip_alignment_1(text_feat)  
            ft_context = torch.cat([query, ft_query_global.unsqueeze(1).repeat(1, n_q, 1), ft_query], dim=-1)
        else:
            ft_context = torch.cat([query, ft_query], dim=-1)  

        sample_traj = []
        for i in range(num_sample):
            pred_traj = self.cvae.inference(ft_context)      
            pred_traj = pred_traj.reshape(n_b, n_q, -1, 3)
            pred_traj = torch.cumsum(pred_traj, dim=-2)       # (B, Q, T-1=4, 3), pos[t] - pos[0]
            pred_traj = pred_traj + query.unsqueeze(-2)       # (B, Q, T-1=4, 3), pos[t]
            sample_traj.append(pred_traj)
        
        sample_traj = torch.stack(sample_traj, dim=2)      
        return sample_traj


@MODELS.register_module()
class ScaleKPST(BaseKPST):
    def __init__(self,
                 loss_args=None,
                 clip_align_args=None,
                 encoder_args=None,
                 decoder_args=None,
                 cvae_args=None,
                 **kwargs):
        super().__init__(loss_args=None, 
                         clip_align_args=clip_align_args,
                         encoder_args=encoder_args, decoder_args=decoder_args,
                         cvae_args=cvae_args)

        if loss_args is not None:
            self.a_weight = loss_args.a_weight
            self.k_weight = loss_args.k_weight
            self.s_weight = loss_args.s_weight
            self.stp_weight = 1.0


    def forward(self, pos, feat, text_feat, dtraj, pack=None):
        if not self.training:
            raise RuntimeError("model needs to be in train mode for training")

        # pos = data['pos']                             # (B, N, 3)
        # feat = data['x']                              # (B, N, 6)
        # text_feat = data['text_feat']                 # (B, F1)
        # dtraj = data['dtraj']                         # (B, Q, T, 3)

        n_b = dtraj.shape[0]
        n_q = dtraj.shape[1]
        query = dtraj[:, :, 0, :]                                # (B, Q, 3)
        ft_text = self.clip_alignment(text_feat)                 # (B, F1)
        
        ft_query = self.get_context(pos, feat, query, n_b, n_q, text_emb=ft_text)  # (B, Q, F2)
        if self.clip_fusion == 'late':
            ft_context = torch.cat([query, ft_text.unsqueeze(1).repeat(1, n_q, 1), ft_query], dim=-1)  # (B, Q, 3+F1+F2)
        elif self.clip_fusion == 'both':
            ft_query_global = self.clip_alignment_1(text_feat)   # (B, F1)
            ft_context = torch.cat([query, ft_query_global.unsqueeze(1).repeat(1, n_q, 1), ft_query], dim=-1)  # (B, Q, 3+F1+F2)
        else:
            ft_context = torch.cat([query, ft_query], dim=-1)  # (B, Q, 3+F1)

        # we must use relative prediction for ScaleKPST
        delta_label = (dtraj[:, :, 1:] - dtraj[:, :, :-1])                         # (B, Q, T-1, 3)
        target_label = (dtraj[:, :, 1:] - dtraj[:, :, 0:1]).reshape(n_b, n_q, -1)  # (B, Q, (T-1)*3)
        pred, sstep_loss, scale_loss, kl_loss = \
            self.cvae(context=ft_context, target=delta_label, pack=pack, return_pred=True)
        pred = torch.cumsum(pred, dim=-2)     # (B, Q, T-1=4, 3), pos[t] - pos[0]
        pred = pred.reshape(n_b, n_q, -1)     # pos[t] - pos[0]
        acc_loss = torch.mean(torch.sum((pred - target_label) ** 2, dim=-1))     
        
        loss = self.stp_weight * sstep_loss + self.s_weight * scale_loss + self.k_weight * kl_loss + self.a_weight * acc_loss 

        loss_pack = {
            "loss": loss,
            "step_loss": sstep_loss,
            "scale_loss": scale_loss,
            "kl_loss": kl_loss,
            "acc_loss": acc_loss,
        }
        return loss_pack

    
@MODELS.register_module()
class ScaleTrajCVAE(nn.Module):

    def __init__(self, 
                 scale_method=None,
                 traj_length=4,
                 vae_args=None,
                 z_scale=1.0,
                 condition_dim=None
                 ):
        super().__init__()
        if (scale_method is not None) and (scale_method not in ['TLN', 'TDN', 'SDN']):
            raise ValueError(f"scale_method need to be in ['TLN', 'TDN', 'SDN'],"
                             f"but get scale_method={scale_method}")

        self.traj_length = traj_length
        self.z_scale = z_scale
        in_dim_traj = traj_length * 3    # (x,y,z)

        if scale_method in ['TLN', 'TDN']:
            in_dim_scale = 1                  # scale for whole trajectory
        elif scale_method in ['SDN']:
            in_dim_scale = traj_length
        else:
            raise ValueError(f"scale_method need to be in ['TLN', 'TDN', 'SDN'],"
                             f"but get scale_method={scale_method}")

        self.scale_method = scale_method
        if vae_args is not None:
            # pdb.set_trace()
            if vae_args.NAME in ['ScaleRegVAE', 'ScaleDiffusionVAE', 'ScaleDiffusionVAEv2']:
                vae_args.in_dim_1 = in_dim_traj
                vae_args.in_dim_2 = in_dim_scale
            elif vae_args.NAME in ['ScaleClsVAE', 'ScaleBinVAE']:
                vae_args.reg_dim = in_dim_traj
                vae_args.cls_dim = in_dim_scale
                if self.scale_method not in ['TLN', 'TDN']:
                    raise ValueError(f"For ScaleClsVAE, scale_method need to be in ['TLN', 'TDN'],"
                                     f"but get scale_method={scale_method}")
            else:
                raise ValueError(f"vae_args.NAME need to be in ['ScaleRegVAE', 'ScaleClsVAE', 'ScaleBinVAE', 'ScaleDiffusionVAE', 'ScaleDiffusionVAEv2'],"
                                 f"but get vae_args.NAME={vae_args.NAME}")
            vae_args.conditional = True
            vae_args.condition_dim = condition_dim
            self.cvae = build_model_from_cfg(vae_args)

            self.latent_dim = self.cvae.latent_dim
        

    def _generate_trajectory(self, pred_step, pred_scale):
        n_b, n_q = pred_step.shape[0], pred_step.shape[1]

        if self.scale_method in ['TLN', 'TDN']:
            # pred_step (B, Q, length*3),  pred_scale (B, Q, 1)
            pred_traj = pred_step * pred_scale
            pred_traj = pred_traj.reshape(n_b, n_q, -1, 3)
        elif self.scale_method in ['SDN']:
            # pred_step (B, Q, length*3),  pred_scale (B, Q, length)
            pred_step = pred_step.reshape(n_b, n_q, -1, 3)
            pred_traj = pred_step * pred_scale.unsqueeze(-1)
        else:
            raise ValueError(f"scale_method need to be in ['TLN', 'TDN', 'SDN'],"
                             f"but get scale_method={self.scale_method}")

        return pred_traj   # (B, Q, length, 3)
    
    
    def _split_trajectory(self, traj, eps=1e-6):
        # traj: (B, Q, length, 3), pos[t] - pos[t-1]
        raise ValueError("EXECUTE _SCALE_SPLIT_TRAJECTORY. There are some problems in dataset preprocess.")
    

    def forward(self, context, target, pack=None, return_pred=False):
        # context: (B, Q, condition_dim)
        # target:  (B, Q, T-1, 3)
        # batch_size = context.shape[0]
        if (pack is not None) and ('ScaleTraj_sstep' in pack.keys()) and ('ScaleTraj_scale' in pack.keys()):
            sstep = pack['ScaleTraj_sstep']
            scale = pack['ScaleTraj_scale']
        else:
            sstep, scale = self._split_trajectory(target)  # (B, Q, F_sstep), (B, Q, F_scale)
        if not return_pred:
            sstep_loss, scale_loss, KLD = self.cvae(sstep, scale, pack=pack, c=context)
            return sstep_loss, scale_loss, KLD
        else:
            pred_step, pred_scale, sstep_loss, scale_loss, KLD = self.cvae(sstep, scale, pack=pack, c=context, return_pred=return_pred)
            pred_traj = self._generate_trajectory(pred_step, pred_scale)
            return pred_traj, sstep_loss, scale_loss, KLD

    def inference(self, context):
        # Semantic-Consistent Latent Varaiable Sampling
        n_b, n_q = context.shape[0], context.shape[1]
        z = torch.randn((n_b, 1, self.latent_dim), dtype=context.dtype, device=context.device)
        z = z.repeat_interleave(n_q, dim=1)

        sstep, scale = self.cvae.inference(z, c=context)
        pred_traj = self._generate_trajectory(sstep, scale)
        return pred_traj


@MODELS.register_module()
class ScaleRegVAE(nn.Module):

    def __init__(self, 
                 in_dim_1, in_dim_2, 
                 hidden_dim, latent_dim, 
                 n_layer_in=1, n_layer_out=1, 
                 n_layer_pred=1,
                 conditional=False, condition_dim=None):

        super().__init__()

        self.latent_dim = latent_dim
        self.conditional = conditional

        if self.conditional and condition_dim is not None:
            input_dim = in_dim_1 + in_dim_2 + condition_dim
            dec_dim = latent_dim + condition_dim
        else:
            input_dim = in_dim_1 + in_dim_2
            dec_dim = latent_dim
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

        self.enc_MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU())
        self.dec_MLP = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ELU())
        
        for i in range(n_layer_in - 1):
            self.enc_MLP.add_module(
                f'linear_{i}', nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
            )
        for i in range(n_layer_out - 1):
            self.dec_MLP.add_module(
                f'linear_{i}', nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
            )

        self.pred_MLP_1 = nn.Sequential()
        self.pred_MLP_2 = nn.Sequential()
        for i in range(n_layer_pred):
            self.pred_MLP_1.add_module(
                f'linear_{i}', nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
            )
            self.pred_MLP_2.add_module(
                f'linear_{i}', nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ELU())
            )
        
        self.pred_MLP_1.add_module('pred_head', nn.Linear(hidden_dim, in_dim_1))
        self.pred_MLP_2.add_module('pred_head', nn.Linear(hidden_dim, in_dim_2))


    def forward(self, x_1, x_2, pack=None, c=None, return_pred=False):
        if self.conditional and c is not None:
            inp = torch.cat([x_1, x_2, c], dim=-1)
        else:
            inp = torch.cat([x_1, x_2], dim=-1)

        h = self.enc_MLP(inp)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)
        z = self.reparameterize(mean, log_var)
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)

        recon_h = self.dec_MLP(z)
        pred_1 = self.pred_MLP_1(recon_h)
        pred_2 = self.pred_MLP_2(recon_h)
        recon_loss_1, recon_loss_2, KLD = self.loss_fn(pred_1, x_1, pred_2, x_2, mean, log_var)
        if not return_pred:
            return recon_loss_1, recon_loss_2, KLD
        else:
            return pred_1, pred_2, recon_loss_1, recon_loss_2, KLD
        
        
    def loss_fn(self, recon_x_1, x_1, recon_x_2, x_2, mean, log_var):
        recon_loss_1 = torch.mean(torch.sum((recon_x_1 - x_1) ** 2 , dim=-1))                   # (B, Q, D)
        recon_loss_2 = torch.mean(torch.sum((recon_x_2 - x_2) ** 2 , dim=-1))                   # (B, Q, D)
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp() , dim=-1))  # (B, Q, D)
        return recon_loss_1, recon_loss_2, KLD
    

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def inference(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)

        recon_h = self.dec_MLP(z)
        pred_1 = self.pred_MLP_1(recon_h)
        pred_2 = self.pred_MLP_2(recon_h)
        return pred_1, pred_2