import torch
import torch.nn as nn
import torch.nn.functional as F

def q_rotate_vector(q, v):
    """
    Rotates vector v by quaternion q.
    q: (B, 4) [x, y, z, w]
    v: (B, 3)
    Formula: v' = v + 2w(q x v) + 2(q x (q x v))
    """
    # Extract parts
    q_vec = q[..., :3] # x, y, z
    q_w = q[..., 3:]   # w

    # Cross products
    t = 2.0 * torch.cross(q_vec, v,dim=-1)
    return v + (q_w * t) + torch.cross(q_vec, t,dim=-1)

def q_inverse(q):
    """Conjugate of unit quaternion is inverse"""
    # q is [x, y, z, w], inverse is [-x, -y, -z, w]
    inv = q.clone()
    inv[..., :3] = -inv[..., :3]
    return inv

# def q_multiply(q1, q2):
#     """
#     Multiply two quaternions.
#     Output is q1 * q2 (rotation q2 followed by q1)
#     """
#     x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
#     x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
#     return torch.stack([x, y, z, w], dim=-1)
def q_multiply(q1, q2):
    """
    Multiply two quaternions. 
    Supports broadcasting (e.g. q1=(N, 1, 4), q2=(1, N, 4) -> Output=(N, N, 4))
    """
    # FIX: unbind(-1) extracts the last dimension components regardless of batch shape
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    
    # Math is now done on tensors of shape (N, 1) and (1, N), which broadcast to (N, N)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)
def deadzone_loss(input_tensor, threshold=20.0):
    """
    Loss is 0 if |x| < threshold.
    Grows quadratically (Smooth) or linearly beyond threshold.
    Using L2 behavior for smoother gradients at the boundary.
    """
    # Magnitude of the vector/scalar
    mag = input_tensor.norm(dim=-1)
    
    # Excess: how much are we outside the safe zone?
    excess = F.relu(mag - threshold)
    
    # Squared error on the excess
    return (excess ** 2).mean()

# class SfMPoseLoss(nn.Module):
#     def __init__(self, 
#                  up_vector=(0, 1, 0),       # HM3D/Habitat usually Y-up
#                  frame0_leash=15.0):         # Radius of the deadzone for Frame 0 (meters)
#         super().__init__()
#         self.register_buffer("up_vector", torch.tensor(up_vector, dtype=torch.float32))
#         self.frame0_leash = frame0_leash

#     def forward(self, pred_t, pred_q, gt_t, gt_q, batch_counts):
#         """
#         pred_t: (N_total, 3)
#         pred_q: (N_total, 4) - Normalized
#         gt_t:   (N_total, 3)
#         gt_q:   (N_total, 4)
#         batch_counts: List[int] - images per episode
#         """
#         total_rel_trans_loss = 0
#         total_rel_rot_loss = 0
#         total_gravity_loss = 0
#         total_scale_loss = 0
#         total_frame0_loss = 0
        
#         cursor = 0
#         num_episodes = len(batch_counts)
        
#         for count in batch_counts:
#             # === SLICE EPISODE ===
#             p_t = pred_t[cursor : cursor+count] # (T, 3)
#             p_q = pred_q[cursor : cursor+count] # (T, 4)
#             g_t = gt_t[cursor : cursor+count]
#             g_q = gt_q[cursor : cursor+count]

#             # --- 1. Frame 0 Constraint (The Leash) ---
#             # We only penalize the first frame if it drifts too far from origin.
#             # This creates the "Flat Zone" you requested.
#             total_frame0_loss += deadzone_loss(p_t[0], threshold=self.frame0_leash)

#             # --- 2. Absolute Gravity Loss ---
#             # We want predicted "Up" to match GT "Up".
#             # World Up Vector (broadcasted)
#             up_vec = self.up_vector.to(p_q.device).unsqueeze(0).expand(count, -1)
            
#             # Rotate World-Up into Local Camera Frame (or vice versa).
#             # Here: Rotate World-Up by Pred Rotation -> Predicted Local Up
#             pred_up = q_rotate_vector(q_inverse(p_q), up_vec)
#             gt_up = q_rotate_vector(q_inverse(g_q), up_vec)
            
#             # Cosine Distance (1 - cos). 
#             # Note: We don't square this because Up is a vector, not a quaternion.
#             # <v1, v2> should be 1.0. 
#             total_gravity_loss += (1.0 - F.cosine_similarity(pred_up, gt_up, dim=-1)).mean()

#             # --- PREPARE RELATIVE POSE ---
#             # To compute relative pose loss, we transform all poses to be relative 
#             # to the First Frame of the episode.
            
#             # 1. Relative Rotation: Q_rel = Q_0_inv * Q_i
#             p_q0_inv = q_inverse(p_q[0:1].expand(count, -1))
#             g_q0_inv = q_inverse(g_q[0:1].expand(count, -1))
            
#             p_q_rel = q_multiply(p_q0_inv, p_q)
#             g_q_rel = q_multiply(g_q0_inv, g_q)
            
#             # 2. Relative Translation, world frame

#             p_t_world = p_t - p_t[0:1]
#             g_t_world = g_t - g_t[0:1]

#             p_t_centered = q_rotate_vector(p_q0_inv, p_t_world) #local to the first frame.
            
#             # Rotate GT by GT_R0_Inv
#             g_t_centered = q_rotate_vector(g_q0_inv, g_t_world)
#             # Rotate translation into Frame 0's coordinates (Optional, but strictly 'relative pose')
#             # If we just want distance consistency, keeping it in world frame aligned to 0 is fine.
#             # Let's keep it in "Zero-Centered World Frame" for simplicity of gradients.
            
#             # --- 3. Scale Alignment (Least Squares) ---
#             # We align p_t_centered to g_t_centered
#             # s = dot(p, g) / dot(p, p)
#             p_flat = p_t_centered.reshape(-1)
#             g_flat = g_t_centered.reshape(-1)

#             # Stop gradient on denominator for stability? 
#             # Usually standard backprop is fine, but clamping is essential.
#             dot_pp = torch.dot(p_flat, p_flat)
#             dot_pg = torch.dot(p_flat, g_flat)
#             dot_gg = torch.dot(g_flat, g_flat)

#             if dot_gg < 1e-4:
#                 print(f"[DEBUG] Found Stationary Episode!")
#                 print(f"GT Translation Norms: {g_t_centered.norm(dim=-1)}")
#                 print(f"Pred Translation Norms: {p_t_centered.norm(dim=-1)}")

#             if dot_gg > 1e-4:
#                 # SAFETY GUARD 2: Check if Model predicts ~0 (Singularity)
#                 if dot_pp < 1e-5:
#                     # Model is effectively outputting 0. 
#                     # We cannot infer scale. 
#                     # Set scale to 1.0 (Identity) so Trans Loss = |0 - GT|.
#                     # This forces the model to grow p naturally via Trans Loss.
#                     scale_raw = 1.0
                    
#                     # We skip the scale loss because log(scale) is meaningless here
#                     scale_loss_term = 0.0
#                 else:
#                     # Normal case
#                     scale_raw = dot_pg / dot_pp
                    
#                     # Log-Safe Loss: log(|s| + eps)
#                     # We add 1e-6 inside abs to prevent log(0)
#                     scale_loss_term = (torch.log(torch.abs(scale_raw) + 1e-6)**2)

#                 total_scale_loss += scale_loss_term
#                 scale = torch.clamp(scale_raw, min=0.001, max=1000.0) # Sanity clamps
#                 p_t_aligned = p_t_centered * scale
#             else:
#                 p_t_aligned = p_t_centered
#             # --- 4. Relative Translation Loss ---
#             # We use Smooth L1. This covers both Direction and Magnitude (Scaled).
#             total_rel_trans_loss += F.smooth_l1_loss(p_t_aligned, g_t_centered, beta=0.1)
            
#             # --- 5. Relative Rotation Loss ---
#             # Double cover check: 1 - <q1, q2>^2
#             dot_prod = (p_q_rel * g_q_rel).sum(dim=1)
#             total_rel_rot_loss += (1.0 - dot_prod**2).mean()

#             cursor += count

#         # Normalize by batch size
#         return {
#             "loss_trans": total_rel_trans_loss / num_episodes,
#             "loss_rot": total_rel_rot_loss / num_episodes,
#             "loss_grav": total_gravity_loss / num_episodes,
#             "loss_leash": total_frame0_loss / num_episodes,
#             "loss_scale": total_scale_loss / num_episodes
#         }

# class SfMPoseLoss(nn.Module):
#     def __init__(self, 
#                  up_vector=(0, 1, 0),       # HM3D/Habitat usually Y-up
#                  frame0_leash=15.0):         # Radius of the deadzone for Frame 0 (meters)
#         super().__init__()
#         self.register_buffer("up_vector", torch.tensor(up_vector, dtype=torch.float32))
#         self.frame0_leash = frame0_leash

#     def forward(self, pred_t, pred_q, gt_t, gt_q, batch_counts):
#         """
#         pred_t: (N_total, 3)
#         pred_q: (N_total, 4) - Normalized
#         gt_t:   (N_total, 3)
#         gt_q:   (N_total, 4)
#         batch_counts: List[int] - images per episode
#         """
#         total_rel_trans_loss = 0
#         total_rel_rot_loss = 0
#         total_gravity_loss = 0
#         total_scale_loss = 0
        
#         cursor = 0
#         num_episodes = len(batch_counts)
        
#         for count in batch_counts:
#             # === SLICE EPISODE ===
#             p_t = pred_t[cursor : cursor+count] # (T, 3)
#             p_q = pred_q[cursor : cursor+count] # (T, 4)
#             g_t = gt_t[cursor : cursor+count]
#             g_q = gt_q[cursor : cursor+count]

#             # --- 2. Absolute Gravity Loss ---
#             # We want predicted "Up" to match GT "Up".
#             # World Up Vector (broadcasted)
#             up_vec = self.up_vector.to(p_q.device).unsqueeze(0).expand(count, -1)
            
#             # Rotate World-Up into Local Camera Frame (or vice versa).
#             # Here: Rotate World-Up by Pred Rotation -> Predicted Local Up
#             pred_up = q_rotate_vector(q_inverse(p_q), up_vec)
#             gt_up = q_rotate_vector(q_inverse(g_q), up_vec)
            
#             # Cosine Distance (1 - cos). 
#             # Note: We don't square this because Up is a vector, not a quaternion.
#             # <v1, v2> should be 1.0. 
#             total_gravity_loss += (1.0 - F.cosine_similarity(pred_up, gt_up, dim=-1)).mean()

#             # --- PREPARE RELATIVE POSE ---
#             # To compute relative pose loss, we transform all poses to be relative 
#             # to the First Frame of the episode.
            
#             # 1. Relative Rotation: Q_rel = Q_0_inv * Q_i
#             g_q0_inv = q_inverse(g_q[0:1].expand(count, -1))
            
#             g_q_rel = q_multiply(g_q0_inv, g_q[1:])
            
#             # 2. Relative Translation, world frame

#             p_t_world = p_t[1:]
#             g_t_world = g_t[1:] - g_t[0:1]

            
#             # Rotate GT by GT_R0_Inv
#             g_t_centered = q_rotate_vector(g_q0_inv, g_t_world)
#             # Rotate translation into Frame 0's coordinates (Optional, but strictly 'relative pose')
#             # If we just want distance consistency, keeping it in world frame aligned to 0 is fine.
#             # Let's keep it in "Zero-Centered World Frame" for simplicity of gradients.
            
#             # --- 3. Scale Alignment (Least Squares) ---
#             # We align p_t_centered to g_t_centered
#             # s = dot(p, g) / dot(p, p)
#             p_flat = p_t_world.reshape(-1)
#             g_flat = g_t_centered.reshape(-1)

#             # Stop gradient on denominator for stability? 
#             # Usually standard backprop is fine, but clamping is essential.
#             dot_pp = torch.dot(p_flat, p_flat)
#             dot_pg = torch.dot(p_flat, g_flat)
#             dot_gg = torch.dot(g_flat, g_flat)

#             if dot_gg < 1e-4:
#                 print(f"[DEBUG] Found Stationary Episode!")
#                 print(f"GT Translation Norms: {g_t_centered.norm(dim=-1)}")
#                 print(f"Pred Translation Norms: {p_t_world.norm(dim=-1)}")

#             if dot_gg > 1e-4:
#                 # SAFETY GUARD 2: Check if Model predicts ~0 (Singularity)
#                 if dot_pp < 1e-5:
#                     # Model is effectively outputting 0. 
#                     # We cannot infer scale. 
#                     # Set scale to 1.0 (Identity) so Trans Loss = |0 - GT|.
#                     # This forces the model to grow p naturally via Trans Loss.
#                     scale_raw = 1.0
                    
#                     # We skip the scale loss because log(scale) is meaningless here
#                     scale_loss_term = 0.0
#                 else:
#                     # Normal case
#                     scale_raw = dot_pg / dot_pp
                    
#                     # Log-Safe Loss: log(|s| + eps)
#                     # We add 1e-6 inside abs to prevent log(0)
#                     scale_loss_term = (torch.log(torch.abs(scale_raw) + 1e-6)**2)

#                 total_scale_loss += scale_loss_term
#                 scale = torch.clamp(scale_raw, min=0.001, max=1000.0) # Sanity clamps
#                 p_t_aligned = p_t_world * scale
#             else:
#                 p_t_aligned = p_t_world
#             # --- 4. Relative Translation Loss ---
#             # We use Smooth L1. This covers both Direction and Magnitude (Scaled).
#             total_rel_trans_loss += F.smooth_l1_loss(p_t_aligned, g_t_centered, beta=0.1)
            
#             # --- 5. Relative Rotation Loss ---
#             # Double cover check: 1 - <q1, q2>^2
#             dot_prod = (p_q[1:] * g_q_rel).sum(dim=1)
#             total_rel_rot_loss += (1.0 - dot_prod**2).mean()

#             cursor += count

#         # Normalize by batch size
#         return {
#             "loss_trans": total_rel_trans_loss / num_episodes,
#             "loss_rot": total_rel_rot_loss / num_episodes,
#             "loss_grav": total_gravity_loss / num_episodes,
#             "loss_scale": total_scale_loss / num_episodes
#         }
class SfMPoseLoss(nn.Module):
    def __init__(self, 
                 up_vector=(0, 1, 0),       
                 frame0_leash=15.0):
        super().__init__()
        self.register_buffer("up_vector", torch.tensor(up_vector, dtype=torch.float32))

    def forward(self, pred_t, pred_q, gt_t, gt_q, batch_counts):
        total_rel_trans_loss = 0
        total_rel_rot_loss = 0
        total_gravity_loss = 0
        total_scale_loss = 0
        
        cursor = 0
        num_episodes = len(batch_counts)
        valid_episodes = 0 # Counter for episodes > 1 frame
        
        for count in batch_counts:
            # === SLICE EPISODE ===
            # Full episode data
            p_t_full = pred_t[cursor : cursor+count]
            p_q_full = pred_q[cursor : cursor+count]
            g_t_full = gt_t[cursor : cursor+count]
            g_q_full = gt_q[cursor : cursor+count]
            
            cursor += count # Advance cursor immediately for safety

            # Edge Case: Single frame episodes cannot have relative pose
            if count < 2:
                continue
            valid_episodes += 1

            # --- PREPARE GT RELATIVE POSE (The Ground Truth Target) ---
            
            # 1. GT Rotation relative to Frame 0
            # Target = q_0_inv * q_i
            g_q0_inv = q_inverse(g_q_full[0:1]) # (1, 4)
            
            # We expand to N-1 because we drop Frame 0
            g_q0_inv_expanded = g_q0_inv.expand(count-1, -1) 
            
            g_q_rel = q_multiply(g_q0_inv_expanded, g_q_full[1:])
            
            # 2. GT Translation relative to Frame 0 (Local Frame)
            # vector = q_0_inv * (t_i - t_0)
            g_t_world_diff = g_t_full[1:] - g_t_full[0:1]
            g_t_local = q_rotate_vector(g_q0_inv_expanded, g_t_world_diff)

            # --- PREPARE PREDICTIONS ---
            # Assumption: Model output at index i (where i > 0) IS the relative pose 0->i.
            # We discard prediction at index 0 entirely.
            p_t_rel = p_t_full[1:]
            p_q_rel = p_q_full[1:]

            # --- 1. Scale Alignment (Least Squares) ---
            p_flat = p_t_rel.reshape(-1)
            g_flat = g_t_local.reshape(-1)

            dot_pp = torch.dot(p_flat, p_flat)
            dot_pg = torch.dot(p_flat, g_flat)
            dot_gg = torch.dot(g_flat, g_flat)

            # Stationary Check
            if dot_gg > 1e-4:
                # Singularity Check
                if dot_pp < 1e-5:
                    scale_raw = 1.0
                    scale_loss_term = 0.0
                else:
                    scale_raw = dot_pg / dot_pp
                    scale_loss_term = (torch.log(torch.abs(scale_raw) + 1e-6)**2)

                total_scale_loss += scale_loss_term
                scale = torch.clamp(scale_raw, min=0.001, max=1000.0)
                p_t_aligned = p_t_rel * scale
            else:
                p_t_aligned = p_t_rel

            # --- 2. Translation Loss ---
            total_rel_trans_loss += F.smooth_l1_loss(p_t_aligned, g_t_local, beta=0.1)
            
            # --- 3. Rotation Loss ---
            dot_prod = (p_q_rel * g_q_rel).sum(dim=1)
            total_rel_rot_loss += (1.0 - dot_prod**2).mean()

            # --- 4. Gravity Loss (Relative) ---
            # We want the "Up" vector in the Local Frame to match.
            # GT Up (Local) = R_rel_gt^T * (0,1,0)
            # Pred Up (Local) = R_rel_pred^T * (0,1,0)
            
            up_vec = self.up_vector.to(p_q_rel.device).unsqueeze(0).expand(count-1, -1)
            
            pred_up_local = q_rotate_vector(p_q_full, up_vec)
            gt_up_local = q_rotate_vector(g_q_full, up_vec)
            
            total_gravity_loss += (1.0 - F.cosine_similarity(pred_up_local, gt_up_local, dim=-1)).mean()

        # Prevent div by zero if all episodes were single-frame
        denom = max(1, valid_episodes)
        
        return {
            "loss_trans": total_rel_trans_loss / denom,
            "loss_rot": total_rel_rot_loss / denom,
            "loss_grav": total_gravity_loss / denom,
            "loss_scale": total_scale_loss / denom
        }
    
class AllPairsPoseLoss(nn.Module):
    def __init__(self, up_vector=(0, 1, 0), frame0_leash=15.0):
        super().__init__()
        self.register_buffer("up_vector", torch.tensor(up_vector, dtype=torch.float32))
        self.frame0_leash = frame0_leash

    def forward(self, pred_t, pred_q, gt_t, gt_q, batch_counts):
        """
        Computes All-Pairs Relative Pose Loss.
        O(N^2) complexity, but strictly invariant to global frame initialization.
        """
        total_loss_trans = 0
        total_loss_rot = 0
        total_loss_grav = 0
        total_loss_scale = 0
        
        cursor = 0
        valid_episodes = 0
        
        for count in batch_counts:
            # === SLICE EPISODE ===
            p_t = pred_t[cursor : cursor+count] # (N, 3)
            p_q = pred_q[cursor : cursor+count] # (N, 4)
            g_t = gt_t[cursor : cursor+count] # (N, 3)
            g_q = gt_q[cursor : cursor+count]
            
            cursor += count
            if count < 2: continue
            valid_episodes += 1

            # === 1. COMPUTE RELATIVE TRANSLATION MATRIX (N, N, 3) ===
            # T_{i->j} = R_i^T * (t_j - t_i)
            # This represents the translation from frame i to j, seen in frame i.
            
            # Diff Matrix: (N, N, 3)
            # diff[i, j] = t[j] - t[i]
            p_diff = p_t.unsqueeze(0) - p_t.unsqueeze(1) 
            g_diff = g_t.unsqueeze(0) - g_t.unsqueeze(1)
            
            # Rotation Inverse (Conjugate) for frame i
            # Expand to (N, 1, 4) to broadcast over j dimension
            p_q_inv = q_inverse(p_q).unsqueeze(1) 
            g_q_inv = q_inverse(g_q).unsqueeze(1)
            
            # Apply Rotation: Result is (N, N, 3)
            p_t_rel = q_rotate_vector(p_q_inv, p_diff)
            g_t_rel = q_rotate_vector(g_q_inv, g_diff)

            # === 2. COMPUTE RELATIVE ROTATION MATRIX (N, N, 4) ===
            # Q_{i->j} = Q_i^{-1} * Q_j
            
            # Broadcast multiply: (N, 1, 4) * (1, N, 4)
            # q_multiply supports broadcasting
            p_q_rel = q_multiply(p_q_inv, p_q.unsqueeze(0))
            g_q_rel = q_multiply(g_q_inv, g_q.unsqueeze(0))

            # === 3. MASKING ===
            # We only care about i < j (future predictions).
            # i == j is 0, i > j is just the inverse (redundant).
            # Create Upper Triangular Mask (offset 1 to exclude diagonal)
            mask = torch.triu(torch.ones(count, count, device=p_t.device, dtype=torch.bool), diagonal=1)
            
            # Flatten to (M, 3) and (M, 4) where M is number of valid pairs
            p_t_flat = p_t_rel[mask]
            g_t_flat = g_t_rel[mask]
            
            p_q_flat = p_q_rel[mask]
            g_q_flat = g_q_rel[mask]

            # === 4. SCALE ALIGNMENT (Global over pairs) ===
            # Find scalar s to minimize sum || s * p_vec - g_vec ||^2
            # s = (p . g) / (p . p)
            
            # Flatten spatial dim for dot product
            p_vec = p_t_flat.reshape(-1)
            g_vec = g_t_flat.reshape(-1)
            
            dot_pp = torch.dot(p_vec, p_vec)
            dot_pg = torch.dot(p_vec, g_vec)
            
            # Stationary Check (on Ground Truth pairs)
            g_mag = torch.dot(g_vec, g_vec)
            
            if g_mag > 1e-4:
                if dot_pp < 1e-5:
                    s = 1.0
                    scale_loss = 0.0
                else:
                    s = dot_pg / dot_pp
                    scale_loss = (torch.log(torch.abs(s) + 1e-6)**2)
                
                # Clamp for stability
                s = torch.clamp(s, min=0.001, max=1000.0)
                total_loss_scale += scale_loss
                
                p_t_aligned = p_t_flat * s
            else:
                p_t_aligned = p_t_flat

            # === 5. LOSSES ===
            
            # Translation (Smooth L1 over all pairs)
            total_loss_trans += F.smooth_l1_loss(p_t_aligned, g_t_flat, beta=0.1)
            
            # Rotation (Cosine over all pairs)
            # 1 - <q1, q2>^2
            rot_dot = (p_q_flat * g_q_flat).sum(dim=1)
            total_loss_rot += (1.0 - rot_dot**2).mean()
            
            # Gravity (Relative)
            # Compare how "Up" transforms from frame i to j
            up_vec = self.up_vector.to(p_q.device).unsqueeze(0).expand(p_q.shape[0], -1)
            
            # Using the absolute quaternions
            p_up = q_rotate_vector(p_q, up_vec)
            g_up = q_rotate_vector(g_q, up_vec)
            
            total_loss_grav += (1.0 - F.cosine_similarity(p_up, g_up, dim=-1)).mean()

        denom = max(1, valid_episodes)
        return {
            "loss_trans": total_loss_trans / denom,
            "loss_rot": total_loss_rot / denom,
            "loss_grav": total_loss_grav / denom,
            "loss_scale": total_loss_scale / denom
        }