import torch.nn as nn
import torch

class CHT:

    def __init__(self, model, sparsity, remove_method, regrow_method, mum_epoch, zeta, chain_removal_list, non_sparse_layer_list, device):
        self.model = model
        self.sparsity = sparsity
        self.remove_method = remove_method
        self.regrow_method = regrow_method
        self.num_epoch = mum_epoch
        self.zeta = zeta
        self.chain_removal_list = chain_removal_list
        self.non_sparse_layer_list = non_sparse_layer_list
        self.device = device

        # get layers dict
        self.linear_layers, self.linear_layers_module = self.find_linear_layers(self.model)

        # get mask dict
        self.mask = {}
        for key in self.linear_layers.keys():
            if key in self.non_sparse_layer_list:
                continue
            self.mask[key] = (torch.zeros_like(self.linear_layers[key]).uniform_(0, 1) >= self.sparsity).to(self.device)

        self.apply_mask_to_weights()

        print("---------------------------------------Non sparse layers---------------------------------------")
        if not self.non_sparse_layer_list:
            print("No non-sparse layers found")
        for name in self.non_sparse_layer_list:
            print(f"{name} -> {self.linear_layers_module[name]}")
        print("---------------------------------------chain removal list--------------------------------------")
        if not self.chain_removal_list:
            print("Chain removal list is empty")
        else:
            for i in range(len(self.chain_removal_list)):
                for name in self.chain_removal_list[i]:
                    print(f"{i}-th chain {name} -> {self.linear_layers_module[name]}")
        print("-----------------------------------------------------------------------------------------------")


    def apply_mask_to_weights(self):
        with torch.no_grad():
            for name in self.linear_layers.keys():
                if name not in self.non_sparse_layer_list:
                    weight = self.linear_layers[name]
                    mask = self.mask[name]
                    weight.data *= mask

    def find_linear_layers(self, model):
        linear_layers = {}
        linear_layers_module = {}
        print("-------------------------------------cht find_layers-------------------------------------")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                linear_layers[name] = module.weight
                linear_layers_module[name] = module
                print(f"find layer: {name} -> {module}")
        return linear_layers, linear_layers_module

    def chts_scores_easy(self, mask, CH_method="CH3"):
        original_shape = mask.shape
        if mask.dim() > 2:
            mask = mask.view(mask.size(0), -1)

        DTPATHS1 = torch.clone(mask)
        TDPATHS1 = DTPATHS1.transpose(1, 0)
        DDPATHS2 = torch.matmul(DTPATHS1, TDPATHS1)
        TTPATHS2 = torch.matmul(TDPATHS1, DTPATHS1)

        BDDPATHS2 = DDPATHS2 != 0
        BTTPATHS2 = TTPATHS2 != 0

        elcl_DT = (torch.sum(DTPATHS1, dim=1) - DDPATHS2) * BDDPATHS2
        elcl_TD = (torch.sum(TDPATHS1, dim=1) - TTPATHS2) * BTTPATHS2

        elcl_DT[elcl_DT == 0] = 1
        elcl_TD[elcl_TD == 0] = 1

        elcl_DT -= 1
        elcl_TD -= 1
        if CH_method == "CH2":
            elcl_DT = 1 / (elcl_DT + 1) * (DDPATHS2 + BDDPATHS2)
            elcl_TD = 1 / (elcl_TD + 1) * (TTPATHS2 + BTTPATHS2)
        elif CH_method == "CH3":
            elcl_DT = 1 / (elcl_DT + 1) * BDDPATHS2
            elcl_TD = 1 / (elcl_TD + 1) * BTTPATHS2
        elif CH_method == "CH3.1":
            elcl_DT = 1 / ((elcl_DT + 1) ** (1 + (elcl_DT / (1 + elcl_DT)))) * (DDPATHS2 + BDDPATHS2)
            elcl_TD = 1 / ((elcl_TD + 1) ** (1 + (elcl_TD / (1 + elcl_TD)))) * (TTPATHS2 + BTTPATHS2)

        elcl_DT = torch.matmul(elcl_DT, DTPATHS1)
        elcl_TD = torch.matmul(elcl_TD, TDPATHS1)

        scores = elcl_DT + elcl_TD.T
        scores = scores * (DTPATHS1 == 0)

        return scores.view(original_shape)
    
    def remove_unactive_links_backward(self, current_adj, after_adj):
        outdegree = torch.sum(after_adj, dim=0)
        # print(current_adj.shape, outdegree.shape)
        # exit()
        outdegree[outdegree > 0] = 1
        current_num = torch.sum(current_adj)
        # print(torch.sum(current_adj, dim=1), torch.sum(current_adj, dim=0))
        # print(torch.sum(torch.sum(current_adj, dim=1) > 0), torch.sum(outdegree))

        current_adj = current_adj * outdegree.reshape(-1, 1)

        # print(torch.sum(torch.sum(current_adj, dim=1) > 0), torch.sum(outdegree))

        print("Number of removed unactive links backwards: ", int(current_num - torch.sum(current_adj)))

        return current_adj

    def remove_unactive_links_forward(self, current_adj, before_adj):
        indegree = torch.sum(before_adj, dim=1)
        indegree[indegree > 0] = 1
        current_num = torch.sum(current_adj)

        # print(torch.sum(torch.sum(current_adj, dim=0) > 0), torch.sum(indegree))
        current_adj = current_adj * indegree.reshape(1, -1)

        # print(torch.sum(torch.sum(current_adj, dim=0) > 0), torch.sum(indegree))

        print("Number of removed unactive links forwards: ", int(current_num - torch.sum(current_adj)))

        return current_adj

    def weight_evolution(self, epoch):

        def chain_prune(weight, zeta, epoch, current_mask, n_prune, n_ones, n_keep, mask1_total):
            n_total = torch.numel(current_mask)
            n_ones_layer = torch.sum(current_mask).item()
            n_prune_layer = int(n_ones_layer * zeta)
            n_keep_layer = n_ones_layer - n_prune_layer

            n_ones.append(n_ones_layer)
            n_prune.append(n_prune_layer)
            n_keep.append(n_keep_layer)

            # create drop mask
            if self.remove_method == "weight_magnitude":
                score_drop = torch.abs(weight)
                _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                new_values = torch.where(
                    torch.arange(n_total, device=weight.device) < n_keep_layer,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices)
                )
                mask1 = new_values.scatter(0, sorted_indices, new_values)

            elif self.remove_method == "weight_magnitude_soft":
                score_drop = torch.abs(weight)
                T = 1 + (2 * epoch / self.num_epoch)

                mask1 = torch.zeros_like(score_drop.view(-1)).to(weight.device)
                flat_matrix = (score_drop.flatten()) ** T
                probabilities = flat_matrix / flat_matrix.sum()
                sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep_layer), replacement=False)
                mask1[sampled_flat_indices] = 1

            elif self.remove_method == "ri":
                eplison = 0.00001
                score_drop = torch.abs(weight) / torch.sum(torch.abs(weight) + eplison, dim=0) + torch.abs(
                    weight) / torch.sum(torch.abs(weight) + eplison, dim=1).reshape(-1, 1)
                _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)

                new_values = torch.where(
                    torch.arange(n_total, device=weight.device) < n_keep_layer,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices)
                )
                mask1 = new_values.scatter(0, sorted_indices, new_values)

            elif self.remove_method == "ri_soft":
                eplison = 0.00001
                score_drop = torch.abs(weight) / torch.sum(torch.abs(weight) + eplison, dim=0) + torch.abs(
                    weight) / torch.sum(torch.abs(weight) + eplison, dim=1).reshape(-1, 1)
                T = 1 + (2 * epoch / self.num_epoch)

                mask1 = torch.zeros_like(score_drop.view(-1)).to(weight.device)
                flat_matrix = (score_drop.flatten()) ** T
                probabilities = flat_matrix / flat_matrix.sum()
                sampled_flat_indices = torch.multinomial(probabilities, max(1, n_keep_layer), replacement=False)
                mask1[sampled_flat_indices] = 1

            else:
                raise NotImplementedError
            mask1 = mask1.float()

            mask1_total.append(torch.reshape(mask1, current_mask.shape))

            return n_ones, n_prune, n_keep, mask1_total


        def chain_regrow(weight, current_mask, layer_index, n_prune, n_ones, mask1_total, name):

            n_prune[layer_index] = int(n_ones[layer_index] - torch.sum(mask1_total[layer_index]))

            print(f"Sparse layer {name}, number of regrowth links: {n_prune[layer_index]}")

            if self.regrow_method == "chts":
                scores = self.chts_scores_easy(mask1_total[layer_index])

                mask2 = torch.zeros_like(scores.view(-1)).to(weight.device)
                flat_matrix = scores.flatten()
                probabilities = flat_matrix / flat_matrix.sum()
                sampled_flat_indices = torch.multinomial(probabilities, max(1, n_prune[layer_index]), replacement=False)
                mask2[sampled_flat_indices] = 1

            elif self.regrow_method == "SET":
                scores = torch.rand(weight.shape).to(weight.device) * (mask1_total[layer_index] == 0)
                thre = torch.sort(scores.ravel())[0][-n_prune[layer_index]]
                mask2 = torch.zeros_like(scores).to(weight.device)
                mask2[scores >= thre] = 1

            elif self.regrow_method == "cht":
                scores = self.chts_scores_easy(mask1_total[layer_index])
                thre = torch.sort(scores.ravel())[0][-n_prune[layer_index]]
                mask2 = torch.zeros_like(scores).to(weight.device)
                mask2[scores >= thre] = 1
            else:
                raise NotImplementedError

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)

            grow_tensor = torch.zeros_like(weight)
            new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            new_weights = torch.where(new_connections.to(weight.device), grow_tensor, weight)

            mask_combined = (mask1_total[layer_index] + mask2_reshaped).bool()

            new_weights *= mask_combined

            return new_weights, mask_combined

        print("-----------------------------------------Sparse training----------------------------------------")

        if not self.chain_removal_list:
            for name in self.linear_layers.keys():
                if name in self.non_sparse_layer_list:
                    continue
                n_ones, n_prune, n_keep, mask1_total = chain_prune(self.linear_layers[name], self.zeta, epoch, self.mask[name], [], [], [], [])
                self.linear_layers[name].data, self.mask[name].data = chain_regrow(self.linear_layers[name], self.mask[name],0, n_prune, n_ones, mask1_total, name)
        else:
            for name in self.linear_layers.keys():
                if name in self.non_sparse_layer_list:
                    continue
                else:
                    if not any(name in row for row in self.chain_removal_list):
                        n_ones, n_prune, n_keep, mask1_total = chain_prune(self.linear_layers[name], self.zeta, epoch,self.mask[name], [], [], [], [])
                        self.linear_layers[name].data, self.mask[name].data = chain_regrow(self.linear_layers[name],self.mask[name], 0, n_prune, n_ones, mask1_total, name)
                    else:
                        continue

            for chain in self.chain_removal_list:
                chain_number = 1
                print(f"-----------------------------------------chain removal {chain_number} ---------------------------------------")
                chain_list = []
                n_prune = []
                n_keep = []
                n_ones = []
                mask1_total = []
                for i in range(len(chain)):
                    chain_list.append(i)
                    n_ones, n_prune, n_keep, mask1_total = chain_prune(self.linear_layers[chain[i]], self.zeta, epoch, self.mask[chain[i]], n_prune, n_ones, n_keep, mask1_total)
                # chain_removal
                for j in reversed(range(len(chain_list) - 1)):
                    mask1_total[chain_list[j]] = self.remove_unactive_links_backward(mask1_total[chain_list[j]], mask1_total[chain_list[j + 1]])
                for j in range(1, len(chain_list)):
                    mask1_total[chain_list[j]] = self.remove_unactive_links_forward(mask1_total[chain_list[j]],  mask1_total[chain_list[j - 1]])
                # chain regrowth
                for i in range(len(chain)):
                    self.linear_layers[chain[i]].data, self.mask[chain[i]].data = chain_regrow(self.linear_layers[chain[i]], self.mask[chain[i]], i, n_prune, n_ones, mask1_total,chain[i])
                chain_number += 1

        print("------------------------------------------------------------------------------------------------")

