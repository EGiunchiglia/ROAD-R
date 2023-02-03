"""

Author: Mihaela C Stoian

"""

"""
This script computes the logic-based regularisation term to be added to the loss during training.
It contains three such regularisation methods, each depending on a diffrent t-norm:
    - <godel_disjunctions_sparse>, for the Godel t-norm
    - <lukasiewicz_disjunctions_sparse>, for the Lukasiewicz t-norm
    - <product_disjunctions_sparse>, for the Product t-norm.
Each of these methods uses a sparse matrix representation to efficiently compute the logic-based loss on a GPU.

For defining a *new t-norm* method, the format of the three methods above should be followed:
e.g. the sparse matrix representation will be kept the same for efficiency, 
but the first step of defining <constr_values> might differ depending on the t-norm: e.g. the matrix might be
initialised with 0s or 1s or something else; next, the functions used inside the for loop will change depending on the
t-norm -- e.g. Godel t-norm used maximum, while Product t-norm used element-wise product; 
finally, the last step -- before taking the mean over <constr_values> -- might again differ, depending on the t-norm;
e.g. for the Lukasiewicz t-norm, we took the element-wise minimum between two matrices, while for the Product t-norm
we negated <constr_values> and added 1 to each element.

Finally, the script contains a method called <logical_requirements_loss>, 
which prompts the computation of one of the 3 methods above, 
depending on the provided t-norm type captured in the <logic> argument.
This method can be modified as more t-norm methods are added to the codebase.
"""

from req_handler import *


def get_size_of_tensor(data):  # in gigabytes
    return data.element_size() * data.nelement() / 1e9


def get_sparse_representation(req_matrix):
    req_matrix = req_matrix.to_sparse()
    return req_matrix.indices(), req_matrix.values()

# Mihaela Stoian
def godel_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    constr_values = torch.zeros(sH.shape[0], NUM_REQ).to(sH.device)

    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] = torch.maximum(
            constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]],
            predictions_at_nnz_values_plus[:, indices_nnz_plus[1] == k])
        constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] = torch.maximum(
            constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]],
            predictions_at_nnz_values_minus[:, indices_nnz_minus[1] == k])

    req_loss = torch.mean(constr_values)
    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
    # and hence we want to minimize the 1-p
    return 1 - req_loss

# Mihaela Stoian
def lukasiewicz_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    constr_values_unbounded = torch.zeros(sH.shape[0], NUM_REQ).to(sH.device)

    # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
    # original sH tensor as well (so grads are updated through sH, which is what we want)
    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values_unbounded[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] += predictions_at_nnz_values_plus[:,
                                                                                     indices_nnz_plus[1] == k]
        constr_values_unbounded[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] += predictions_at_nnz_values_minus[
                                                                                       :,
                                                                                       indices_nnz_minus[1] == k]

    constr_values = torch.min(torch.ones_like(constr_values_unbounded), constr_values_unbounded)
    req_loss = torch.mean(constr_values)

    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements, and hence we want to minimize the 1-p
    return 1 - req_loss

# Mihaela Stoian
def product_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    # The disjunction is more complex to implement thant the conjunction
    # e.g., A and B --> A*B while A or B --> A + B - A*B
    # Thus we see the disjunction as the negation of the conjunction of the negations of all its
    # literals (i.e., A or B = neg (neg A and neg B))

    constr_values = torch.ones(sH.shape[0], NUM_REQ).to(sH.device)

    # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
    # original sH tensor as well (so grads are updated through sH, which is what we want)
    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] *= predictions_at_nnz_values_plus[:,
                                                                           indices_nnz_plus[1] == k]
        constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] *= predictions_at_nnz_values_minus[:,
                                                                             indices_nnz_minus[1] == k]

    # Negate the value of the conjunction
    req_loss = torch.mean(1. - constr_values)

    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
    # and hence we want to minimize the 1-p
    return 1 - req_loss


def logical_requirements_loss(preds, labels, num_pos, alpha, gamma, logic, Cplus, Cminus):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''

    # Discard all the labels we are not interested in
    H = preds[:, 1:NUM_LABELS + 1]  # discard agentness class

    if len(H) == 0:
        req_loss = torch.zeros(1).cuda().squeeze()
        return req_loss

    # Since we have replicated now we have that the matrices are 3-dims tensors where dimension 0 has len 1
    # --> we need to unsqueeze the tensors to get back the original matrices
    # Iplus, Iminus = Iplus.squeeze(), Iminus.squeeze()
    # Mplus, Mminus = Mplus.squeeze(), Mminus.squeeze()
    Cplus, Cminus = Cplus.squeeze(), Cminus.squeeze()

    req_loss = torch.zeros([1]).cuda()

    if logic == "Godel":
        req_loss = godel_disjunctions_sparse(H, Cplus, Cminus)
    elif logic == "Lukasiewicz":
        req_loss = lukasiewicz_disjunctions_sparse(H, Cplus, Cminus)
    elif logic == "Product":
        req_loss = product_disjunctions_sparse(H, Cplus, Cminus)
    else:
        print("Cannot be here, logic {:} not defined".format(logic))
        exit(1)

    return req_loss

