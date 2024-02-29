> [!NOTE]
> **Paper summaries to refresh in memory.**

# CV: localization (detection, segmentation)

**Wang, Jiaqi, Kai Chen, Shuo Yang, Chen Change Loy, and Dahua Lin. 2019. “[GA-RPN] Region Proposal by Guided Anchoring.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1901.03278.**

The authors propose a novel method GA-RPN (guided anchoring) for object detection. Let we have a multi-level feature pyramid from FPN. For each level we first predict a probability map that should indicate the center of some object  (denoted as N_L at fig. 1). To train this stage, authors sort all ground truth objects by feature pyramid levels (fig. 2), define positive a small rectangular region inside object's box (green), and do not apply loss for a larger region inside object's box (yellow). The rest of the space is defined as negative (gray). This gives a ground truth annotation for the probability map, and authors apply focal loss due to high positive-negative imbalance.

Secondly, for each level and spatial location we predict log-height and log-width (denoted as N_S at fig. 1). This differs from a regular box regression, since the center of the object is not shifted. For the center point the ground truth W and H are trivial, and for other points the ground truth W and H are defined using formula 5. Then authors add a feature adaptation stage consiting of deformable convolution (denoted as N_T at fig. 1). Finally, we select spatial positions from a probability map based on some threshold, this gives us a set of region proposals and an adapted feature map.

Experientally, the proposed method gives higher recall with lower number of proposals. This method can be applied to any pretrained two-stage object detector by replacing RPN with GA-RPN and re-training RoI head. Also this method can be applied to single-stage detectors, where the sliding window anchoring scheme is replaced with the proposed guided anchoring. Visualization of results can be seen on fig. 4, 7.

**Kirillov, Alexander, Ross Girshick, Kaiming He, and Piotr Dollár. 2019. “Panoptic Feature Pyramid Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1901.02446.**

Authors complain that every competitive entry in the panoptic challenges used separate networks for instance and semantic segmentation, with no shared computation. They propose a unified architecture Panoptic FPN for panoptic segmentation. This is a modification of Mask R-CNN, when a semantic segmentation branch is attached to FPN (fig. 3). This branch nearly copies the structure of the FPN, adding new top-down connections, and can be viewed as an assymetric, lightweight decoder (comparing to a symmetric decoder in U-Net, fig. 5). The authors argue that high quality semantic segmentation requires high-resolution, rich, multi-scale features - identify exactly the characteristics of FPN.

Authors train their model simultaneuosly on semantic segmentation and instance segmentation, and show it is a robust and accurate baseline for both tasks.
 
**Li, Yanghao, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. 2019. “Scale-Aware Trident Networks for Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1901.01892.**
 
The authors propose TridentNet for object detection. Given an input image, a usual ResNet backbone processes it 3 times, with dilation rates 0, 1 and 2 in 3x3 convolutions. This can be seen as 3 branches with shared weights, but different dilation rates (fig. 2, 3). Then, RPN head and RoI head with shared weights prosesses each of the three outputs. Each dilation rate is assoclated with some object scale range, and region proposals not in the range are filtered out during training and inference. At inference time, we can either run all the 3 branches, concatenate predictions and apply NMS, or run only the second branch with small performance decrease.

The proposed architecture addresses the problem when Image Pyramid (fig. 1a) achieves true scale invariance while being slow, and FPN (fig. 1b) is faster but region features of objects with different scales are extracted from different levels of FPN backbone, which in turn are generated with different sets of paramers.

IMO, TridentNet and Image Pyramid are quite similar, since in both architectures the backbone is applied several times with shared parameters. Dilated 3x3 convolution can be approximated by downsampling + 3x3 convolution + feature upsampling. In comparison with Image Pyramid, TridentNet cannot be used with an arbitrary pretrained backbone, since it use a modified backbone architecture.
 
**Lu, Xin, Buyu Li, Yuxin Yue, Quanquan Li, and Junjie Yan. 2018. “Grid R-CNN.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1811.12030.**
 
The authors propose Grid R-CNN object detector, which is a modification of Faster R-CNN that uses a different way to refine bounding boxes in RoI head. Authors define 3x3 relative positions (4 corner points, midpoints of 4 edges and the center point) and apply several dilated convolutions and deconvolutions to RoI-aligned features to predict 9 masks of size 56x56 (fig. 2) for each RoI. Each mask corresponds to some relative position and is supervised by a heatmap when a cross of 5 points in the corresponding point is set to 1 and all other points are set to 0. During inference, the predicted heatmaps are projected back to the original image. For example, to predict the upper boundary of a bounding box, authors use weighted average of y-axis coordinates of the three upper grid points, using predicted probabilities as weights. To be able to predict boundaries outside the RoI (i. e. to expand the RoI) authors extend the representation region of the feature map (fig. 4).

On COCO benchmark, the proposed approach gives 4.1% AP gain at IoU=0.8 and 10.0% AP gain at IoU=0.9, while slightly decreasing AP at IoU=0.5. The categories with the most gains usually have a rectangular or bar like shape (e.g. keyboard, laptop, fork, train, and refrigera- tor), while the categories suffering declines or having least gains usually have a round shape without structural edges (e.g. sports ball, frisbee, bowl, clock and cup).

IMO, the approach is very similar to R-FCN, it seems like both these works use the same approach, differing only in details. Also AttractioNet and CornerNet are quite similar, since they also do not rely on box regression. If out goal is to predict boxes very approximately, it may be enough to use box regression, which is some form of long-range prediction. For example, based on the size of the head, we can approximately predict the size of the whole human. However, if our goal is precise box prediction, we probably should switch to heatmaps, like done in this work.

IMO, the task of precise box regression is overrated, since in practice we often need to either just count objects, or segment objects, but not to detect objects with high box quality. However, even in the task of object counting, accurate box prediction can serve as intermediate step, so that the correct boxes are not suppressed by the boxes of other objects during NMS. To solve this problem we either need to predict more accurate boxes by using things like Grid R-CNN or Repulsion loss, or switch to NMS-free detectors, like Relation networks.

IMO, the future of object detectors are generative ones, when predictions are used to "explain" feature maps (the same as in the case when points are explained by mixture of multivariate normal distributions, as usual in EM-algorithm). If feature map is explained by one predicted object, it no longer requires additional explanations, so no more objects are predicted in this place, eliminating the need for NMS.
 
**Li, Buyu, Yu Liu, and Xiaogang Wang. 2018. “Gradient Harmonized Single-Stage Detector.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1811.05181.**
 
The authors propose GHM-C loss for binary (foreground/background) classification in single-stage object detectors and GHM-R loss for bounding box regression. These losses aim to harmonize the contribution of easy and hard examples and outperform focal loss in most metrics. Collaborate with GHM, the performance of single-stage detector can surpass state-of-the-art two-stage detectors like FPN and Mask-RCNN with the same ResNext backbone. The motivation and formulation are described below.

The gradient of cross-entropy w.r.t. logit equals P-P*, where P is the predicted probability and P* is the ground truth label. The gradient norm G = |P-P*| can be considered as the contribution of the example to the learning process. The distribution of G across all training samples for a converged one-stage detection model is plotted on fig. 2. The easy negatives have a dominant number and have a great impact on the global gradient. Moreover, we can see that a converged model still can’t handle some very hard examples ("outliers"). Authors hypothesize that if the converged model is forced to learn to classify these outliers better, the classification of the large number of other examples tends to be less accurate.

Consider i-th training samle with gradient norm G\_i. The authors propose to calculate the gradient density, which is the ratio of examples lying in the small-size region centered at G\_i (that is y-axis value on fig. 2). GHM-C loss works in the following way: the cross-entropy loss value i-th sample is divided by the calculated gradient density for this sample (formulas 3-8). This can be seen as dynamic re-weighting training samples: the examples with large density are relatively down-weighted. With the GHM-C loss, the huge number of very easy examples are largely down-weighted and the outliers are slightly down-weighted as well, which simultaneously addresses the attribute imbalance problem and the outliers problem. Further, to make training more stable, authors propose to use exponential moving average (EMA) to smooth gradient density values.

For bounding box regression authors propose GHM-R loss, which works in the similar way and uses dynamic weights based on gradient norm from bouding box regression loss. To make gradient computations more robust, authors propose to replace a regular Huber (smooth L1) loss (formula 15), which is essentially a gradient clipping, by authentic smooth L1 (ASL1) loss (formula 17). Further, gradient density and dynamic weights are calculated w.r.t. to ASL1 loss.

IMO, this paper lacks visualizations of classification outliers. They can be either wrong annotations or examples when "shortcuts" do not work (see the paper "Shortcut Learning in Deep Neural Networks"). In the latter case, down-weighting them may hurt performance in the presence of the distributional shift.
 
**Law, Hei, and Jia Deng. 2018. “CornerNet: Detecting Objects as Paired Keypoints.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1808.01244.**
 
Authors propose CornerNet one-stage object detector (fig. 4). As a backbone authors use two stacked hourglass networks, taking final feature map from the last layer. For each spatial location, CornerNet predicts 3 outputs: (i) if this location is a top-left corner of any object, (ii) if this location is a bottom-right corner of any object, and (iii) embedding vector for the object, that is used for pairing top-left and bottom-right corners. The model is trained to "pull" corner embeddings for the same object and "push away" corner embeddings for different objects. So, the actual values of the embed-dings are unimportant, only the distances between the embeddings are used to group the corners. To produce tighter bounding boxes, the network also predicts offsets to slightly adjust the locations of the corners, so a corner cannot be localized based on local evidence, and some non-local operation is needed.

As a neck, authors use corner pooling which take the maximum values in two directions, each from a separate feature map, and add the two maximums together (fig. 3). The need for this operation is explained as follows: a corner of a bounding box is often outside the object.

Authors hypothesize two reasons why detecting corners would work better than bounding box centers or pro-posals. First, the center of a box can be harder to localize because it depends on all 4 sides of the object, whereas locating a corner depends on 2 sides and is thus easier, and even more so with corner pooling, which encodes some explicit prior knowledge about the definition of corners. Second, corners provide a more efficient way of densely discretizing the space of boxes: we just need O(w*h) corners to represent O(w^2*h^2) possible anchor boxes.
 
**Redmon, Joseph, and Ali Farhadi. 2018. “YOLOv3: An Incremental Improvement.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1804.02767.**
 
Authors present YOLOv3, which have minor modifications comparing with YOLOv2. Authors use sigmoid instread of softmax for multi-class classification, since sometimes classes overlap (e. g. "woman" vs "human" in Google Images dataset). Also authors use new ResNet-like backbone with good accuracy-speed tradeoff, which is the aim of the whole work.

While not achieving SOTA, YOLOv3 runs significantly faster than RetinaNet with comparable performance. It may be good at counting the number of zebras in a national park, or tracking someone's cat as it wanders around their house.
 
**Cheng, Bowen, Yunchao Wei, Honghui Shi, Rogerio Feris, Jinjun Xiong, and Thomas Huang. 2018. “Revisiting RCNN: On Awakening the Classification Power of Faster RCNN.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1803.06799.**
 
Authors propose to augment Faster R-CNN with additional correctional classifier (fig. 4) that crops boxes from the image, passes them to CNN and classifies them, without sharing backbone with Faster R-CNN and without propagating gradient from correctional classifier to Faster R-CNN.

Authors argue that this may solve the problem when classification needs translation invariant feature, whereas  localization needs translation covariant feature, and the proposed solution allows to decrease the count of hard false positives (fig. 3). IMO, too complex argumentation for my brain.
 
**Liu, Shu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. 2018. “[PANet] Path Aggregation Network for Instance Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1803.01534.**
 
Authors propose PANet, which is a modification of Mask R-CNN for object detection and instance segmentation. Firstly, authors add another bottom-up path to FPN (fig. 1b). While the red line on fig. 1 consists of 100+ layers, the green "shortcut" line consists of less than 10 layers. Authors hyphothesize that this solution "enhances the localization capability of the entire feature hierarchy by propagating strong responses of low-level patterns". As a result, the backbone produces feature pyramid denoted N2, …, N5.

Given a region proposal, authors perform adaptive feature pooling (fig. 1c, fig. 6). This operation performs RoIAlign from all levels of feature pyramid, firstly process each region incependently, then fuse them all with elementwise max operation. It turns out that 50%+ of the features are pooled from lower levels. "This observation clearly indicates that features in multiple levels together are helpful for accurate prediction. It is also a strong support of designing bottom-up path augmentation."

Finally, authors add a head for classification and box prediction (fig. 1d) and a custom head for mask prediction (fig. 1e, fig. 4). Ablation studies justify standalone usefulness of all described modifications.

IMO, the whole backbone is another example when where we add short connections while keeping long ones, which boosts performance (another examples are ResNet, DenseNet, RNN with attention and transformer). The PANet backbone can actually be re-drawn as U-Net + additional stack of layers. As for adaptive feature pooling, "lower levels" and "higher levels" doesn't look like a very appropriate terms here, since all these layers may contain semantically strong features that came from the higher layers P4-P5.

**Kirillov, Alexander, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. 2018. “Panoptic Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1801.00868.**

Authors propose a panoptic segmentation task where the goal is to segment both "thing" instances (countable objects such as people, animals, tools) and "stuff" areas (amorphous regions of similar texture or material such as grass, sky, road). To make panoptic annotation, one should first choose a set of thing and stuff classes. Each pixel of an image must be assigned a class label and an instance id, when for "stuff" classes instance id is ignored, and instance order does not matter. Ambiguos or out-of-class pixels are labeled as "void". Such annotation format is a strict generalization of the format for semantic segmentation, and does not allow overlapping segments, which are allowed in instance segmentation. Authors note that humans are not always consistent in panoptic segmentation tasks: some inconsistencies are shown in figures 3, 4.

Authors propose a panoptic quality (PQ) metric. To calculate this metric, we first need a predictions in the same format as annotations. We associate predicted and ground truth segments that have mask IoU > 0.5. Authors prove that this IoU threshold produces unique matching: each ground truth segment can have at most one corresponding predicted segment with IoU > 0.5 and vice versa. Then for each class, the unique matching splits the predicted and ground truth segments into three sets: true positives (TP), false positives (FP), and false negatives (FN). PQ metric is a multiplication of segmentation quality (average IoU) and recognition quality terms (fig. 2, formula 2).

Authors propose a baseline that leverages pretrained models for instance segmentation and semantic segmentation. Instance segmentation models may produce overlapped segments, so NMS-like procedure is applied. We iterate over sorted instances, starting from the most confident. For each instance we first remove pixels which have been assigned to previous segments, then, if a sufficient fraction of the segment remains, we accept the non-overlapping portion, otherwise we discard the entire segment. Then, we combine instance segments with semantic segmentation results by resolving any overlap between thing and stuff classes in favor of the thing class (this heuristic is imperfect but sufficient as a baseline).

Authors note that in the pre-deep learning era there was interest in the joint semantic+instance segmentation task described using various names such as scene parsing, image parsing, or holistic scene understanding. however later the schism between semantic and instance segmentation has led to a parallel rift in the methods for these tasks: stuff classifiers are usually built on FCN with dilations while object detectors often use object proposals and are region-based.

IMO, "scene parsing" seems inappropriate term here, because "parsing" sounds like inferring 3D world from the image, and a perfect "image parsing" also requires amodal completion, depth estimation, keypoint segmentation, fine-grained classification etc.

**Cai, Zhaowei, and Nuno Vasconcelos. 2017. “Cascade R-CNN: Delving into High Quality Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1712.00726.**

Authors propose Cascade R-CNN model for object detection (fig. 3d) that extends Faster R-CNN (fig. 3a) with cascade of sequential RoI heads. Firstly, RPN (denoted as H0, omitted in fig. 3d) performs foreground-vs-background classification (C0) and box regression (B0), thus returning a set of proposals. Next, the first head (H1, C1, B1) performs box regression for each proposal to refine its coordinates. Then, refined box proposals are RoI pooled again and passed to the second head (H2, C2, B2), and so on (3 heads in total). During training, box proposals are passed through stages, while each stage has it's own IoU requirement (between predicted and target boxes): 0.5, 0.6, and 0.7 for three stages, respectively. On each stage, we filter box proposals that do not fit the IoU requirements (termed "outliers" on fig. 2).

On inference, we can use outputs from any head. Authors find that outputs from the third head give the highest AP50-95 on MS COCO test set. If we add 4th head, AP50-95 slightly decreases. while AP90 continues to improve (fig. 4).

Authors provide a detailed justification for Cascade R-CNN architecture. When training a bounding box regressor, we usually use IoU hyperparameter to divide boxes into positive (which should be corrected to target boxes) and negative (which should be ignored). Authors found that training IoU matters. Let we train two box regressors: one with IoU=0.5 and another with IoU=0.7. It turns out that the first regressor is good at correcting very inaccurate proposals, but even degredes already accurate proposals. The second regressor is better than the first on accurate proposals, but is much worse on inaccurate proposals (fig. 1a). Authors conclude that each regressor has its own specialization and a single regressor can only be optimal for a single quality level.

(IMO, this conclusion doesn't seem 100% justified, at least we could try to perform box regression in another way, by binning the output space and output probabilities instread of regressing single numbers, as done in AttractioNet. Also, we can try constrained box regression, while each stage cannot perform a large box shifts and can perform only small ones)

A vanilla Faster R-CNN is depicted on fig. 3a. Some works have argued that a single box regression step is insufficient for accurate localization, and applied several regression steps iteratively (fig. 3b). Here the same head is used for each step. This ignores the above problem that a regressor trained at IoU=0.5 is suboptimal for box proposals of higher IoUs (even degrades them), and vice versa. While boxes become mor accurate after fist stages, subsequent stages may degrade them. Due to these problems, iterative box regression requires a fair amount of human engineering, in the form of proposal accumulation, box voting, etc., and has somewhat unreliable gains. Usually, there is no benefit beyond applying the same box regressor twice.

One possible solution is to train a box regressor with multiple IoUs by using integral loss, which is a combination of several losses with different IoU thresholds (formula 6). But the problem here is that different losses operate on different numbers of positives: the set of positive samples decreases quickly with increasing IoU threshold, so we may face overfitting for high-IoU loss terms. Such approach has very little gain over vanilla Faster R-CNN.

This naturally leads to Cascade R-CNN design, when we train a separete box regressor for each refinement stage. The regressors of the deeper stages are optimized for higher IoU thresholds and accept boxes from previous stage, that are filtered by IoU during training. There is no overfitting here, since examples are plentiful at all levels.

**Hu, Han, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. 2017. “Relation Networks for Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1711.11575.**

Authors propose a relation modules, which replace RoI head and NMS stage in two-stage object detectors.

1. Instance recognition stage. A relation module accepts as input RoI-pooled feature vectors and bounding boxes from several RoIs, and is essentially a self-attention between feature vector, with one modification: in addition to feature vectors, it uses bounding box coordinates. A usual attention calculates softmax(attn\_logits), which is an L1-normalization of exp(attn\_logits). In relation module, exp(attn_logits) is additionally multiplied by scores that reflect relative position or bounding box pairs (formula 5). Authors use several attention heads and concatenate their outputs (fig. 2). So, a relation module can be viewed as Multi-Head Self Attention (MHSA) with additional usage of bounding box information. In contrast to transformers ("Attention is all you need"), position-wise MLP is not applied after each MHSA layer, and instead fully-connected layer is applied before the first layer and after first several layers. After the last layer, a linear layer is applied to calculate scores and perform bounding box regression (formula 10).
2. Duplicate removal stage. Authors replace NMS with modified version of relation modules (fig. 3, right). As well as region embeddings and boxes, these modules also use scores from the previous stage. These scores are sorted in descending order, obtaining ranks, which are then embedded into a higher dimensional feature vector and added to region embeddings obtained from the previous stage. Further details see in sec. 4.3.

As a result, authors obtain the first end-to-end object detector without NMS post-processing stage. It may seem like end-to-end training presents a problem, because the goals of instance recognition step and duplicate removal step seem contradictory (the former expects all objects matched to the same ground truth object to have high scores, but the latter expects only one of them does). Nevertheless, authors found the end-to-end training works well.

Authors hypothesize that the proposed approach can be applied to other tasks such as instance segmentation, action recognition, object relationship detection, caption, VQA, etc.

**Liu, Songtao, Di Huang, and Yunhong Wang. 2017. “Receptive Field Block Net for Accurate and Fast Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1711.07767.**

Based on neuroscience studies, authors propose Receptive Field Block (RFB) by concatenating outputs of multiple convolutions with different weights and dilation rates (fig. 2). In fig. 3, authors compare RFB with Inception block, atorous spatial pyramid pooling and deformable convolution. Authors incorporate RFB in SSD detectior with a lightweight backbone, getting RFB Net: an object detector with high quality and computational efficiency.

**Wang, Xinlong, Tete Xiao, Yuning Jiang, Shuai Shao, Jian Sun, and Chunhua Shen. 2017. “Repulsion Loss: Detecting Pedestrians in a Crowd.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1711.07752.**

Authors propose a Repulsion Loss (RepLoss) for box regression in object detection (formula 1). It adds two additional terms to a regular (L1 or IoU) loss. Let we have a predicted box and an associated target box. The first term (RepGT) penalizes IoU between predicted box and other targets (fig. 1), that could effectively stop a predicted bounding box from shifting to its neighboring objects which are not its target. The second term (RepBox) penalizes IoU between the current predicted box and predicted boxes with different associated targets. This term is able to reduce the probability that the predicted bounding boxes with different regression targets are merged into one after NMS.

In the results of RepBox, there are fewer predictions lying in between two adjacent ground-truths, which is desirable in crowd scenes in pedestrian detection (fig. 7). Also, RepLoss improves mAP on PASCAL VOC detection.

**Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. “[RetinaNet] Focal Loss for Dense Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1708.02002.**

Authors propose a focal loss for classification head in one-stage object detectors. This loss is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases (fig. 1). This scaling factor can automatically focus the model on hard examples during training. So, the focal loss performs the opposite role of a robust loss (for example, Huber loss) that reduce the contribution of outliersby down-weighting the loss of examples with large errors. Authors describe the focal loss for binary classification, while extending it to multi-class classification is straighforward. Focal loss have two hyperparameters: alpha and gamma (see formula 5 when p_t is defined by formula 2 and alpha_t is described in 3.1). When alpha=0.5 and gamma=0, focal loss reduces to the logloss.

Focusing on hard examples is important for one-stage object detectors because of high foreground-vs-background class imbalance: these detectors evaluate 10-100 thousands of candidate locations per image but only a few locations contain objects. Also, to increase stability in early training, authors propose changes in initialization (sec. 3.3).

Authors propose a RetinaNet model for object detection, which incorporates focal loss and have anchor box classification and regression heads (for 9 anchors) attached to each level of FPN on top of ResNet architecture (fig. 3). As for focal loss, gamma=2 works well in practice and the RetinaNet is robust to gamma from 0.5 to 5. Authors report SOTA on MS COCO detection, surpassing even two-stage architectures.

IMO, focal loss is a controversial solution. In my experiments for few-shot object detection with YOLOv5, applying focal loss affects a confidence interval that yields good F1 measure on a test set. Focal loss makes this interval (i) narrow, (ii) dataset-dependent, and (iii) epoch-dependent (usually shifts to the right during training). Hence, we usually cannot be satisfied with threshold 0.5 and need a large enough validation set to select the threshold, which is not accessible in few-shot learning. Just to mention, focal loss is disabled by default in YOLOv5.

**He, Kaiming, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. 2017. “Mask R-CNN.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1703.06870.**

Authors propose a model Mask R-CNN to perform simultaneous object detection and instance segmentation. Comparing to vanilla Faster R-CNN, it has several changes.

Firstly, RoIPool is replaced by RoIAlign. Just like RoIPool, it takes a feature map and region of interest and returns a feature map of fixed size, but it has no qualization and instead use bilinear interpolation (fig. 3). This boosts performance even for object detection, but for segmentation it has even larger effect.

To predict masks, authors add another branch to RoI head (that accepts fixed-size feature map from RoIAlign). It consists of conv and deconv layers and outputs 80 masks corresponding to 80 MS COCO classes (fig. 4). The model is trained on both object detection and instance segmentation annotations. For mask predictions, authors use sigmoid instead of softmax over classes, because we already have a brach to predict a class for an object, so there is no need to predict class for every mask point. Loss is applied only on ground truth class mask. Authors note that if segmentation head returns only one class-agnostic mask, this decreases performance just for a little bit. However, replacing sigmoid with softmax over classes decreases performance very significantly.

Authors train two variants of Mask R-CNN with two different backbones. First variant (ResNet-50-C4) uses ResNet as a backbone, and feature map is obtained from C4 layer, when the 5-th stage of ResNet (which is compute intensive) is moved into RoI head (fig 4., a). Output mask has a size of 14x14 for each RoI. Second variant (ResNet-50-FPN) uses Feature Pyramid Network (FPN) as a backbone, concatenates features from multiple FPN outputs, has a lighter RoI head and outputs mask of size 28x28 for each RoI (fig 4., b). Second variant is less compute and memory intensive, so it allows backpropagating through more RoI at each step: "each image has N sampled RoIs, with a ratio of 1:3 of positive to negatives. N is 64 for the C4 backbone and 512 for FPN". FPN version yields better quality than C4. ResNeXt-101-FPN backbone achieves the best quality overall (table 2a).

Mask R-CNN outperforms another competitive models with multi-scale train/test, horizontal flip test, and online hard example mining (OHEM). Authors expect such improvements to be beneficial also for Mask R-CNN, leaving this work for the future.

Also, authors extent Mask R-CNN framework to predict human keypoints by training mask head to predict a mask for each keypoint. In this case, for each keypoint mask the target is only one pixel corresponding to the position of the keypoint, and authors apply softmax over spatial axes.

**Dai, Jifeng, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. 2017. “Deformable Convolutional Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1703.06211.**

Authors propose a deformable convolution layer (fig. 2, 5). For simplicity, consider a specific filter size 3x3 and C output channels. In a regular 3x3 convolution we have 9 relative spatial locations: R = {(-1, 1), (-1, 0), ..., (1, 1)}. In contrast, a deformable 3x3 convolution filter is two-stage. At the first stage, we apply a regular convolution with 2*9 output channels and obtain float-point (x, y) offsets for each relative location in R. It worth noting that these offsets are calculated independently for each spatial location.

At the second stage, we apply a convolution with C output channels, while adding offset for each relative location in R. This gives us fractional offsets, and to handle them we apply bilinear interpolation to input feature map. This can be seen as sampling positions from feature map with float-point offsets and then applying a regular convolution. Both stages are learnable and differentiable, so are easy to integrate into any CNN architectures

Also, authors propose a deformable RoI pooling (fig. 4) that works the same way. As well as in deformable convolution, this allows to select spatial locations at runtime for each image.

In fig. 6, 7, authors visualize selected offsets for deformable convolution and selected regions for deformable RoI pooling. An initial motivation was better robustness to geometrical transformations, and authors demonstrate the superiority of their methods for DeepLab, class-aware RPN, faster R-CNN and R-FCN models.

**Fu, Cheng-Yang, Wei Liu, Ananth Ranga, Ambrish Tyagi, and Alexander C. Berg. 2017. “DSSD : Deconvolutional Single Shot Detector.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1701.06659.**

This is a concurrent work with FPN, when authors also aim to combine low-level, but spatially accurate features from low levels of CNN, and high-level features from high layers, to better detect small objects. Proposed architecture is very similar to FPN, but a module that combines features from different layers is more complex. It uses deconvolution, batch normalization and combines features with product instead of sum, because the experimental results show that the element-wise product provides the best accuracy (fig. 1, 2, 3). As in SSD, when training, each layer is responsible for some scale of anchor boxes.

**Redmon, Joseph, and Ali Farhadi. 2016. “[YOLOv2] YOLO9000: Better, Faster, Stronger.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1612.08242.**

Authors propose YOLOv2 model for object detection. While first version of YOLO predicts two boxes per spatial location and use bipartite matching to apply loss, YOLOv2 predicts 5 or 9 anchor boxes per spatial location, as in RPN. But in contrast to RPN, anchor box sizes and aspect ratios were obtained automatically  from MS COCO by K-means clustering, which gives 5% mAP gain comparing to hand-crafted boxes. Authors say that K-means anchor boxes allow to largely increase maximum recall, while slightly decrease mAP, comparing to no anchors and bipartite matching. Secondly, authors propose to increase feature map resolution from 13x13 to 26x26 by concatenating features from two last layers. Finally, YOLOv2 has new backbone with batch normalization and is trained on varying resolutions from 320 to 608. All improvements and metric gains are listed in table 2.

In addition, authors propose YOLO9000 (as an extension for YOLOv2) that was trained simultaneously for detection and image-level classification and is able to detect >9000 classes. They take 9000 classes from ImageNet and combine these classes with MS COCO classes by using "WordTree" hierarchical classification scheme (fig. 6). where each class has a list of parents, starting from "physical object". At each spatial location YOLO9000 predicts scores for classes and their parents. Given an image with image-level class C, we search for a predicted box with the highest score for this class and apply classification and objectness losses to this box, so this is a form of weak supervision. Authors note that YOLO9000 struggle to localize objects that are completely different from MS COCO classes: "COCO does not have bounding box label for anytype of clothing, only for person, so YOLO9000 struggles to model categories like “sunglasses” or “swimming trunks”.

**Lin, Tsung-Yi, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2016. “[FPN] Feature Pyramid Networks for Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1612.03144.**

Authors propose Feature Pyramid Network (FPN), see fig. 1 (d) as a backbone for obect detection. Comparing to a regular CNN backbone, FPN contains top-down connections, which are implemented as nearest neighbor upsampling, 1x1 conv and addition (see fig. 3). This is very similar to SharpMask model for segmentation from the same authors.

FPN returns multiple feature maps of different spatial resolution. Then, RPN and RoI pooling heads from Faster R-CNN are attached to all feature maps with shared parameters. Faster R-CNN based on FPN achieves both good quality and good FPS on detection task.

Fig. 1 compares FPN with different alternatives. Fig. 1 (b) is an object detection head on top of CNN encoder. To increase quality, we can scale an image to different sizes and apply CNN + detection head to each size (a), but this is slow. Instead, we could attach detection head to different CNN layers (c), but authors complain that in this case first feature maps will be produced by pretty shallow CNN encoder, so are not "semantically strong". We could add more CNN layers before the first feature map, as done in SSD architecture, but thus we either need a lot of FLOPS (if there is no pooling layers) or the first feature map will already have low spatial resolution. FPN is aimed at solving these problems.

IMO, FPN closely resembles U-Net, up to some minor implementation details, such as addition instead of concatenation in top-down path. Authors do compare with U-Net, noticing that in FPN predictions are made independently at all levels.

**Zhang, Liliang, Liang Lin, Xiaodan Liang, and Kaiming He. 2016. “Is Faster R-CNN Doing Well for Pedestrian Detection?” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1607.07032.**

Authors develop a network for object detection based on RPN plus boosted forest. Interesting in this work is that by default RPN performs well as a stand-alone pedestrian detector, but the downstream classifier degrades the results. Authors say that this is because (i) insufficient resolution of feature maps for handling small instances which can lead to “plain” features caused by collapsing bins, and (ii) lack of any bootstrapping strategy for mining hard negative examples.

**Gidaris, Spyros, and Nikos Komodakis. 2016. “[AttractioNet] Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1606.04446.**

Authors propose AttractioNet: an iterative scheme to generate region proposals for two-stage object detectors. Given an image feature map and some region, the object location refinement module first expands the region, then performs bilinear pooling (similar to RoI pooling) to obtain a feature map of the fixed size, processes it and and predicts 2M probabilities. Each of these probabilities means whether the specific row or column belongs to the object bounding box (fig. 3). This allows to refine box coordinates and can be viewed as a form of bounding box regression. Also, for each region the objectness score is predicted (does this region enclose some object or not).

Starting from the set of "seed" regions, uniformly distributing across the image, we apply the described operations several times (fig. 4) to get the final region proposals.

With AttractioNet and a VGG16-Net based detector authors report the detection performance on COCO that significantly surpasses all other VGG16-Net based detectors while even being competitive with a heavily tuned ResNet-101 based detector. Also, AttractioNet generalizes to unseen categories to a certain extent.

**Chen, Liang-Chieh, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. 2016. “DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1606.00915.**

Authors propose a DeepLab model for semantic segmentation. This model is based on a simple baseline when CNN directly converts image to class probabilities for each spatial position, and logloss is applied. Authors propose several modifications for this baseline.

Firstly, authors propose atrous convolution layer (fig. 2). Comparing to the regular convolution, atorous convolution has larger receptive field, where some spatial positions are "disabled" (are constant zero). This gives large receptive field with a relatively small number of trainable weights. Comparing to pool + conv layer combination, atorous convolution also enlarges receptive field, but gives higher-resolution feature map (fig. 3). However, it is requires more computations, so authors don't rely solely on them, and add with bi-linear interpolation layers for upsampling (fig. 1).

Secondly, authors propose Atrous Spatial Pyramid Pooling (ASPP) (fig. 4). Instead of running the whole model for an image resized to different sizes, authors apply atorous convolution of different rate to feature map, and merge predictions.

Thirdly, authors propose to refine predictions with Conditional Random Field (CRF). This operation accepts the original image and the output probability map as inputs, defines an energy function (formulas 2, 3) and iteratively minimizes it. The first term in (3) forces pixels with similar color and position to have similar labels, while the second term enforces smoothness. More information on CRFs is available in the paper "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials" (2012). Examples can be seen in fig. 6, 11. 

Also, authors note that they use CRF only as post-processing, while some other works pursue joint learning of the CNN and CRF, by either unrolling the CRF mean-field inference steps to convert the whole system into an  end-to-end trainable network, or by approximating one iteration of the dense CRF mean field inference by convolutional layers with learnable filters.

**Dai, Jifeng, Yi Li, Kaiming He, and Jian Sun. 2016. “R-FCN: Object Detection via Region-Based Fully Convolutional Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1605.06409.**

Authors propose two-stage network for object detection, when RoI network is shallow and contains only one not trainable layer. Usually shallow RoI networks fail to perform well (see sec. 1), so authors develop a special scheme with interpretable feature map. There are 3 heads attached to the backbone: (i) region proposal network, (ii) score map, and (iii) box regression head. All these heads are run once per image. Score map contains k*k*(C+1) output channels (fig. 2), when C is the number of classes and k=3. So, for each class score map returns 3x3 scores, when each score is associated with some object corner (top-left, bottom-center, center-center etc.). For example, if some spatial location contains a top-left corner for a ground truth box of class "car", then we want a corresponding cell in score map to be activated.

Score map is computer wor the whole image. Given a region proposal, we apply a "position-sensitive RoI pooling", that is illustrated on fig. 3. For each class we do the following. Firstly, we perform average pooling from "top-left" score map and top-left area of a region proposal. Then, we perform average pooling from "top-center" score map and top-center area of a region proposal, and so on. As a result, for each class we get 3x3 scores. Then we average them (denoted as "vote" on fig. 3) and get final score for a class and a region proposal.

So the whole RoI head is made from shallow and not learnable operations with negligible per-RoI computation. Nearly the same operation is performed for box regression head (do not described in details here).

All network is trained end-to-end, so, we do not apply loss to the score map and instead propagate a gradient through RoI head. Also, authors propose to adopt online hard example mining (OHEM) for training, because negligible per-RoI computation enables nearly cost-free example mining.

This approach closely resembles InstanceFCN instance segmentation model from the same authors. In comparison, no segmentation annotation is used here, but the resulting score maps (fig. 3, 4) are noisy and not suitable for segmentation task, so R-FCN is not shown to be effective as box-supervised segmentation method.

**Li, Ke, and Jitendra Malik. 2016. “Amodal Instance Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1604.08202.**

Authors generate training data for amodal instance segmentation in the following way. They start from the dataset annotated with modal instance segmentation annotations. Then they crop random objects by their masks and paste on another objects, partially occluding them. Then they pose the following ML task: given a modal bounding box, the task is to predict amodal mask and bounding box.

Since initial masks may also be partially occluded by another objects, authors mark pixels belonging to another objects as "unknown": no loss is applied for these points (blue color on fig. 2). Authors claim that object may be occluded by another object, but not by background (IMO it's debatable).

Authors use the following pipeline: given a modal bounding box, a pretrained frozen model produce modal segmentation heatmap, and then trainable model predicts the amodal segmentation mask and bounding box in an iterative fashion using a new strategy that will be referred to as Iterative Bounding Box Expansion (see details in sec. 4).

**Shrivastava, Abhinav, Abhinav Gupta, and Ross Girshick. 2016. “[OHEM] Training Region-Based Object Detectors with Online Hard Example Mining.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1604.03540.**

Authors propose a method called online hard example mining (OHEM) to incorporate hard sample mining to SGD to fight foreground-vs-background class imbalance in object detection task (in object-proposal-based detectors imbalance may be 70:1, while in detectors based on sliding windows imbalance may be 100K:1).

In OHEM, given a set of region proposals, we calculate loss for each proposal, take N highest losses and backpropagate them. However, there is a small caveat: co-located RoIs with high overlap are likely to have correlated losses. To deal with this, we perform NMS before selecting RoIs with the highest losses. To save computations and memory, we calculate losses for all RoIs in no-grad mode, then after sorting losses we again perform forward pass, then backward pass.

OHEM achieves higher test mAP than both (i) random sampling foreground and background region proposals in some ratio, like in Fast R-CNN, and (ii) training the network on all region proposals, that is calculating and backpropagating RoI head losses for each region proposal.

IMO, the problem of foreground-vs-background class imbalance is not justified enough in this work. Probably it is metric-dependent: should we face it when using FP+FN (a close analogue of accuracy) instead of mAP as metric?

**Dai, Jifeng, Kaiming He, Yi Li, Shaoqing Ren, and Jian Sun. 2016. “[InstanceFCN] Instance-Sensitive Fully Convolutional Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1603.08678.**

Authors propose InstanceFCN: a fully-convolutional architecture for class-agnostic instance segmentation. At each spatial position InstanceFCN predicts 9 classification scores corresponding to 9 relative positions (top-left, top-center, center-center etc.). A classifier should predict true if there is some foreground object in this position, and this position relates to a specified relative position of this object. Also, an objectness score is predicted for each spatial position. At inference time, all predictions are assembled to produce instance segmentation (fig. 4).

**Stewart, Russell, and Mykhaylo Andriluka. 2015. “End-to-End People Detection in Crowded Scenes.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1506.04878.**

https://vc.ru/ml/571586-raspoznavanie-tovarov-na-polkah

**Liu, Wei, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. 2015. “SSD: Single Shot MultiBox Detector.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.02325.**

Authors proposed SSD architecture for object detection. This is very silimar to YOLO (see fig, 1, 2), but instead of one feature map, authors apply detection head to several feature maps from several last layers of CNN encoder. Also, authors perform classification for each anchor box, instead of each spatial position in YOLO.

Later, the idea of producing several feature maps with a single backbone forward pass will come to Feature Pyramid Network (FPN) architecture.

**Huang, Lichao, Yi Yang, Yafeng Deng, and Yinan Yu. 2015. “DenseBox: Unifying Landmark Localization with End to End Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1509.04874.**

Authors propose a model DenseBox for object detection. Spatial resolution of CNN outputs are increased by upsampling, and for each spatial location (corresponding to 4x4 pixels) model peforms bounding box regression and object-vs-background classification (for a single class "face"). For classification, the ground-truth region is a circle inside ground truth bounding box.

IMO, this is very similar to YOLO. One difference is that YOLO use bipartite matching, while DenseBox uses groud truth mask for object-vs-background classification. Comparing with YOLO, authors say that "Our DenseBox uses up-sampling layers to keep a relative high-resolution output, with a down-sampling scale factor of 4 in our model. This enables our network capable to detect very small objects and highly overlapped objects, which YOLO is unable to deal with."

**Zhu, Yan, Yuandong Tian, Dimitris Mexatas, and Piotr Dollár. 2015. “Semantic Amodal Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1509.01329.**

Authors pose a task of amodal segmentation: segmenting the full extent of objects, not just visible pixels (fig. 2). Authors ask humans to annotate regions in images amodally and find that this is a well-posed annotation task: the agreement between independent annotators is high. Naturally, annotations are most consistent for regions with simple shapes and little occlusions. On the other hand, when the object is highly articulated and/or severely occluded, annotators tend to disagree more (fig. 9). So, the authors collect an MS COCO subset of 5000 images with amodal annotations.

As a baselines, authors propose two ways: (i) to take modal mask from another model and output an amodal mask, or (ii) to directly predict amodal masks from image patches. Authors evaluate models for varying occlusion levels q: none (q=0), partial (0<q≤ .25), and heavy (q>.25) and summarize metric in table 3.

**Pinheiro, Pedro O., Ronan Collobert, and Piotr Dollar. 2015. “[DeepMask] Learning to Segment Object Candidates.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1506.06204.**

This is the concurrent work with Faster R-CNN, authors here also aim to extract object proposals for two-stage object detectors. The proposed architecture and training data is shown in fig. 1. Given an image patch, VGG encodes in into a feature map. Classification head predicts if the patch contains any object that is located approximately in the center and has a size of approximately 1/2 of the whole patch. If such object exists, segmentation head predicts it's mask (otherwise no loss is appplied to the segmentation head, so it is free to predict anything, which is good for generalization to unseen categories). The segmentation head starts with 1x1 convolution, then goes 14x14 convolution without activation and finally goes classification output of size 56x56. Last two layers are linear, so they are low-rank version of a single fully connected layer.

During inference, the model is applied densely at multiple locations and scales. For each scale, VGG can be applied once, since it's fully convolutional. Ane example output can be seen in fig. 2.

IMO, the connection between DeepMask and RPN (from Faster R-CNN paper) is interesing. While RPN performs several classifications and box regressions for each spatial position, DeepMask performs a classification and outputs 56x56 mask, but requires segmentation annotations for training. The weights from the last segmentation layer could be visualized as 512 grayscale patches of size 56x56. Probably, these patches could be even made predefined and untrainable. Also, we could add box regression on top of the output segmentation mask, obtaining something similar to RPN.

The recent "Segment Anything" model (2023) is somewhat similar, because it can also return a mask given some point as prompt, but instead of convolution with fixed-size kernel it is based on attention, therefore it is more flexible.

**Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2015. “[YOLO] You Only Look Once: Unified, Real-Time Object Detection.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1506.02640.**

Authors propose YOLO network for object detection, which resembles region proposal network (RPN) in Faster R-CNN, but performs multi-class classification. An image can be divided into grid of spatial locations, and YOLO predicts B bounding boxes for each grid cell (B is a hyperparemeter, by default B=2). Each box is predicted as 1) box regression (x, y, w, h), 2) confidence score and 3) class probabilities (summed to 1), which are the same for all B boxes in a grid cell. So, given N classes, for each grid cell YOLO predicts 4B + B + N numbers.

In contrast to RPN, YOLO does not use anchor boxes. "We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall." Such runtime matching between set or list of predictions and set of ground truths is called bipartite matching (see more details in DETR paper, section 2.3).

After assigning ground truth boxes to predicted boxes, we can apply losses for box regression, confidence and classification, when confidence ground truth value is IoU between predicted and ground truth boxes.
Comparing to OverFeat, authors say that "OverFeat … is still a disjoint system. Over-Feat optimizes for localization, not detection performance. … the localizer only sees local information when making a prediction. OverFeat cannot reason about global context". Comparing to Fast and Faster R-CNN, authors say that "While they offer speed and accuracy improvements over R-CNN, both still fall short of real-time performance."

IMO, YOLO architecture is almost identical with RPN: the only difference between them is the presence of multi-class classification and a way to apply losses. Also, the problem with YOLO is that it cannot distinguish between box coordinates uncertainty and object presence uncertainty (confidence is a combination of both uncertainties). Also it would be interesting to train 1x1 conv YOLO output layer on the top of the frozen pretrained backbone, because it is a good way to check the ability of the backbone to disentangle scale and object class, because both scale and class will be produced as a dot product, so are related to some directions in embedding space.

**Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. 2015. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1506.01497.**

Authors propose Faster R-CNN network for object detection. They augment Fast R-CNN with Region Proposal Network (RPN), that is used for first stage: extracting region proposals (fig. 2). RPN accepts feature map from backbone as input, consists of 3x3 conv layer and two output 1x1 conv heads (actually they can be viewed as one): classification head and regression head (fig. 3). So, RPN and ROI head share the same backbone.

For each spatial location in RPN output, we associate k "anchor boxes": rectangular regions of 3 different scales and 3 different aspect ratios, centered in this location. There are typically 2400 spatial positions in RPN, and so 2400x9 anchors. Classification head predict a score for each position and anchor box, and regression head performs bounding box regression for each position and anchor box.

During training, every anchor box is either marked positive and associated with some ground truth box, or marked as negative, or marked as ambigous (see 3.1.2 for details). RPN learns to classify positively anchors marked as positive and classify negatively anchors marked as negative. For anchors marked as positive, RPN learns to perform bounding box regression to refine it's coorinates. When training RPN, each mini-batch arises from a single image, where the sampled positive and negative anchors have a ratio of up to 1:1.
Authors propose several ways to train the whole model (see 3.2): (i) alternating training, when we alter between training RPN stage and training RoI head stage, and (ii) approximate joint training with multi-task loss. Joint training is called "approximate" because RoI pooling layer by default is not differentiable w.r.t. the box coordinates, but still achieves good results.

IMO, RPN network is equivalent to single-class YOLO, so 1) in single-class scenario RPN can be used as final predictor supplemented by NMS stage, 2) in multi-class scenario RPN's classifier may be made multi-class, getting YOLO as a result. So, a drawback of this paper is absence of ablation studes, when in single-class detection tasks only RPN is used without ROI pooling. See also paper "Is Faster R-CNN Doing Well for Pedestrian Detection?". Also it could be interesting to check if combining box scores from RPN and RoI head can benefit.

**Oquab, Maxime, Léon Bottou, Ivan Laptev, and Josef Sivic. 2015. “Is Object Localization for Free? - Weakly-Supervised Learning with Convolutional Neural Networks.” In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 685–94.**

Authors train a weakly-supervised image recognition model. During training, only image-level annotations are used in form of binary vector where i-th element indicate the presence of i-th class. So, we don't use any information about the number and position of objects of this class. During inference, for each class present in the image, a model should return coordinates (x, y) pointing to any object of this class (fig. 6).

To achieve this, author's model returns an array of class logits for each spatial position of size (W, H, N), when (W, H) is a size of spatial grid (32x less than size of image) and N is the number of classes. After that, sigmoid is applied to get multi-class classification for each spatial position. Then global max pooling is applied, thus we get (i) multi-class image-level classification, and (ii) coordinates for each class. Finally, logloss is applied to resulting image-level class probabilities. Interestingly, for negative images thiscan be seen as hard-negative mining, when we select a point that is the most confusing to the model and optimize it. On inference, such model may return points or even masks for every class (fig. 1).

Authors train model on PASCAL VOC and MS COCO without box annotations. In addition, they rendomly rescale images, instead of costly multi-scale training. It turns out that the quality of the resulting model is not far from R-CNN trained with box supervision in terms of localization (specifying some point), but is much worse in terms of bounding box quality. In authors' experiments, introducing box supervision with masked pooling instead of global pooling doesn't help too much.

Authors say that a problem with this approach is the possibility to classify based on background: "Fore example, the presence of a baseball field is a strong indicator for presence of a baseball bat and a baseball glove." (see also paper "Object Recognition with and without Objects"). IMO, another problem is that if we imagine a perfect model that, given a car, will always classify as "car" some point above the car, it will achieve ideal loss on author's training task, but will still localize incorrectly.

**Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 2015. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1505.04597.**

Authors propose a model U-Net for semantic segmentation (fig. 1), which combines feature maps from different layers (instead of combining final predictions, as done in FCN paper). In U-Net, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture. To train on small amount of biomedical segmentation data, authors use augmentations like cropping and elastic deformations.

**Gidaris, Spyros, and Nikos Komodakis. 2015. “Object Detection via a Multi-Region & Semantic Segmentation-Aware CNN Model.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1505.01749.**

Authors propose several modifications for Fast R-CNN network for object detection. With their methods, authors report SOTA on PASCAL VOC detection.

Firstly, authors propose new RoI pooling method. Given a region proposal, they perform max pooling (that is finding maximum activation inside region for each filter in feature map) for several sub-regions and surrounding regions (fig. 3) and concatenate resulting vectors. IMO, this is not very different from SPP.

Secondly, authors train semantic segmentation FCN backbone (from paper "Fully Convolutional Networks for Semantic Segmentation") in a weakly-supervised fashion, when ground truth annotations for semantic segmentation are obtained by just full filling of ground truth boxes. After such training model is able to predict meaningful object masks (fig. 5). Then segmentation head is removed and we freeze the model. Given a region proposal, we perform max pooling for this region from FCN feature map and concatenate resulting vector to the vector from multi-region CNN.

Thirdly, authors propose to iteratively refine boxes on inference time. The idea is that since RoI head is able to refine box coordinates, we can further use refined boxes as new box proposals, and so on. After performing N such steps we take boxes and their scores from each step and merge them: B = B_1 U B_2 U … U B_n. Each box has it's score from RoI head. Then we perform NMS and obtain subset of boxes: B' = NMS(B). Finally, authors propose to perform box voting: for each b' in B', we search for all similar boxes in B (with IoU > 0.5) and calculate weighted average of these boxes with their scores as weights. This way we get final predictions.

IMO, looking at fig. 6 with false positive dection, it looks very easy  to determine that the car extends beyond the left margin of the box and to reject the box based on this observation. The inability to do this indicates some problems in the objective or inference scheme. Probably, iterative box proposals updates will shift the box leftwards, so we should take the boxes from the last stage and not from all stages. The idea of iterative box regression was further developed in AttractioNet (from the same authors), Cascade R-CNN and DiffusionDet models.

**Girshick, Ross. 2015. “Fast R-CNN.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1504.08083.**

Autor proposed Fast R-CNN architecture for object detection, that is a modification of SPP-Net (which is a modification of R-CNN). Fast R-CNN differs from SPP-Net by collapsing 3 training stages (training classifier head, training SVM and training bouding box regression) into one stage with multi-task loss. 

Author introduces a term "RoI (region-of-interest) pooling" for extraction a fixed-length feature vector from the feature map. RoI pooling is a simplification of SPP: instead of using multiple pooling levels, it crops region of interest from the feature map, splits into 7x7 spatial cells, performs max pooling in each cell and concatenates results (see section 2.1). It can be viewed as "resize+flatten", bug when resizing is performed using max operation instead of average. Whole architecture is depicted on fig. 1.

Importantly, RoI pooling has no interpolation (in contrast to RoIAlign and RoIWarp from subsequent works). RoI pooling first quantizes a floating-number RoI to the discrete granularity of the feature map, this quantized RoI is then subdivided into spatial bins which are themselves quantized, and finally feature values covered by each bin are aggregated (usually by max pooling).

**Ren, Shaoqing, Kaiming He, Ross Girshick, Xiangyu Zhang, and Jian Sun. 2015. “Object Detection Networks on Convolutional Feature Maps.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1504.06066.**

This is a concurrent work with Fast R-CNN (authors also overlap). Paper is about designing object detection subnetworks ("Networks on Convolutional feature maps", NoCs), which are applied to fixed-size feature maps (fig. 1) produced by RoI pooling (see Fast R-CNN). As NoCs, authors try MLPs, ConvNets and ConvNets with maxout. Results suggest that a deeper region-wise NoC is useful and is in general orthogonal to deeper feature maps, and convolutional NoCs are more effective than MLPs.

Authors show that naive version of Faster R-CNN with MLP has low accuracy, because its region-wise classifier is shallow and not convolutional. On the contrary, a deep and convolutional NoC is an essential factor for Faster R-CNN + ResNet to perform accurate object detection.

**Long, Jonathan, Evan Shelhamer, and Trevor Darrell. 2014. “[FCN] Fully Convolutional Networks for Semantic Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1411.4038.**

In this work pre-trained CNN is fune-tuned for pixelwise prediction for semantic segmentation. Firstly, authors view fully-connected layers  as 1x1 convolutions. Authors combine outputs from different layers by using trainable upsampling (initialized by bilinear upsampling) and addition (fig. 3). Thus authors try to combine high-level semantic imformation with low-level precise spatial information, achieving SOTA performance at that time on PASCAL VOC segmentation.

Later, U-Net architecture will be based on FCN, while combining feature maps instead of predictions.

**He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2014. “[SPP-Net] Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1406.4729.**

Authors propose to adapt Spatial Pyramid Pooling (SPP) method from classic computer vision to CNN architectures, as an alternative to global pooling. This is intended to solve cropping vs warping problem in image classification (fig. 1) and to allow training and inference with variable-size images. SPP (fig. 3) is a concatenation of global pooling (avg or max), global 2x2 pooling and global 4x4 pooling. If we use average pooling, it actually can be viewed as "flatten" operation applied to feature map resized to spatial sizes 1x1, 2x2 and 4x4.

This work is also important for object detection (fig. 5). Given region proposals in R-CNN, it's a natural idea to crop regions in feature map, instead of cropping image regions. At first, this will boost speed, secondly it will give more context information (for example, we want to use context information when detecting document parts or very small objects). A question is what operation we want to apply to feature map regions.

IMO, if we apply "flatten" operation, we are limited to a certain region size. If, instead, we apply global pooling operation, we are facing the following problem. Let we have several identical objects placed in a row with bounding boxes B1, …, Bn. Let we perform average pooling on feature map for these boxes, obtaining embeddings e1, …, en. Let also we have a box proposal containing two objects: B1+B2. It's embedding will be (e1+e2)/2. So, e1 and e2 should be classified as object and (e1+e2)/2, (e1+e2+e3)/3 etc. should be rejected. We see that such classes are not linearly separable, and if e1=e2, this task is even unsolvable. SPP method seems to solve this problem.

**Sermanet, Pierre, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. 2013. “OverFeat: Integrated Recognition, Localization and Detection Using Convolutional Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1312.6229.**

Authors propose a method OverFeat for object detection task. Given some image and square region, authors train CNN to perform classification and bounding box regression (which means regress 4 numbers corresponding to 4 sides of a square). In principle, we could apply this CNN to each possible square region, but this is resource-intensive. Instead (thanks to translation equivariance of CNNs) we can extract CNN feature map only 6 times for 6 different scales, obtaining 6 feature map of size 9x9x1024, and then apply classification and regression head to different regions on the feature maps, using sliding window of size 5x5 and flatten operation. On inference time, such model will return a lot of bounding boxes, and we apply NMS.

OverFeat is inferior in quality to R-CNN proposed in the same year, so R-CNN became more widespread. IMO, flatten layer applied to each 5x5 region can be viewed as 5x5 convolution, so in principle this model is not very different from YOLO published 2 years later.

**Girshick, Ross, Jeff Donahue, Trevor Darrell, and Jitendra Malik. 2013. “[R-CNN] Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1311.2524.**

Authors propose two-stage R-CNN architecture for object detection task.

Stage 1. The pipeline starts from extracting ~2 thousands of box proposals for an image. In this work, this stage is not trainable and is performed by selective search algorithm from classic computer vision. This algorithm is based on detecting areas with similar texture and iteratively merging them.

Stage 2. For each box proposal, we crop corresponding image region (with padding) and resize it to 227x227. The resulting image is fed into CNN encoder and classifier, which classifies a region into N object classes + "background" class. We train a classifier to classify as background any crop with IoU < 0.5 with all known ground truth boxes. After training, authors replace classifier with N linear SVMs. one for each class. These SVMs are trained the same way as previous classifier, but use IoU threshold 0.3 instead of 0.5, authors argue that this boosts quality. After training, SVMs can be represented as fully connected layer on top of CNN encoder. Also, R-CNN uses bounding box regression to refine the boundaries of object's box and NMS for box filtering.

Authors initialize backbone with ImageNet-pretrained network before training on object detection.

Comparing to OverFeat, R-CNN have higher quality but is much slower, as it requires multiple CNN forward passes for each image. Later, Fast R-CNN and Faster R-CNN will achieve better performance.

# CV: few-shot

**Sariyildiz, M. B., Kalantidis, Y., Larlus, D., & Alahari, K. (2020). Concept Generalization in Visual Representation Learning. arXiv, 2012.05649. Retrieved from https://arxiv.org/abs/2012.05649v2**

A benchmark ImageNet-CoG for concept generalization, a task which means learning a parameter-efficient (linear) classifier to classify novel classes. Benchmark defines a set of seen concepts along with sets of unseen concepts that are semantically more and more distant from the seen ones.  It is designed to evaluate models pretrained on IN-1K out-of-the-box and draws unseen concepts from the rest of the IN-21K dataset. Authors compare several models. To learn new concepts authors use from 1 to 128 samples, so it is a few-shot learning task.

A question: if we have a known class A and similar new class B, then splitting A to A1, ..., An during pretraining will help? (or predictng not inly the presence of A but also its file-grained properties)

# CV: Image editing

**Geng, Z., Yang, B., Hang, T., Li, C., Gu, S., Zhang, T., ...Guo, B. (2023). InstructDiffusion: A Generalist Modeling Interface for Vision Tasks. arXiv, 2309.03895. Retrieved from https://arxiv.org/abs/2309.03895**

The authors propse to modify the InstructPix2Pix architecture to enable it to solve not only image editing tasks, but also segmentation and keypoint detection. More specifically, the authors propose to unify computer vision tasks as image editing task following a natural language instruction. This contrasts to the previous approaches (Pix2Seq2, Unified IO) where computer vision tasks were unified as seq-to-seq tasks on tokens.

For segmentation, the latent diffusion model performs denoising (starting form the latent noise) while conditioning on the image and on the natural language instruction like "apply a blue semi-transparent mask to the rightmost dog while maintaining the remainder unaltered". This allows to formulate segmentation as image editing task. Since the model returns RGB image, an additional postprocessing step (performed by a lightweight U-Net) is applied to extract mask. For keypoint detection, the conditioning instuction looks like "Please use red to encircle the left shoulder of the man”. The output image should exhibit a red circle at the corresponding location. This allows to formulate keypoint detection as image editing task.

The training stages are as follows (fig. 2). Step 1 is a regular pretraining for a diffusion model. During step 2 (see sec. 3.3), the model adapts to the domain of images with colored masks or circles placed in random positions. This allows the model to adapt to such images, which substantially deviate from typical natural images. During this stage, the original image caption is augmented with a suffix, such as ”with a few different color patches here and there” or ”surrounded with a red circle”, and this stage is performed in a self-supervised fashion.

Step 3 is a task-specific training, when segmentation and keypoint detection datasets (like MS COCO) are used to produce inputs, targets and instructions. The model is also trained on deblurring, denoising, and watermark removal task, formulated as editing instructions. To enhance the diversity of instructions, the authors first manually write 10 instructions for each task, then use GPT-4 to rewrite and expand the diversity of these instructions, thereby mimicking user input to the system. Authors also collect their own dataset for object removal and editing, based on SA-1B and OpenImages (see sec. 3.2) and use it for training. Finally, the autors use web crawling to collect 23000 image editing examples performed by humans using Photoshop. These data are also used for training and benchmarking. The final step 4 is human alignment, when the model is fine-tuned on its own cherry-picks selected by humans.

Authors note that the trained model is able to generalize to keypoints not present in the training data (for example, animals' body parts and car logos). Also the model turns out to be able to solve classification and detection tasks, if we formulate them as segmentation and post-process outputs to profuce classes and bounding boxes. This, authors conclude that the model exhibits AGI capabilities to some extent, being able to generalize to tasks not seen during training. IMO, this is too strong of a statement, since the model actually performs the same task (segmentation) , just another post-processing is applied.

IMO, a big shortcoming of this work is that recognition tasks (segmentation, keypoint detection) are performed as iterative diffusion process, that is pretty slow. While for image editing tasks this may be OK (since humans also take a lot of time to draw pictures), for image recognition tasks this seems a bad approach. U-Net inside diffusion models should probably extract high-level imformation from the image to denoise it, so we need to find a way to obtain this information in one forward pass (or, maximum, in 2-3 passes). For example, ODISE model is designed in this way.

**Brooks, Tim, Aleksander Holynski, and Alexei A. Efros. 2022. “InstructPix2Pix: Learning to Follow Image Editing Instructions.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2211.09800.**

The authors fine-tune Stabele Diffusion v1.5 (based on latent diffusion architecture) to perform instructional image editing, that is mapping from original image and editing instruction (like "Make his jacket out of leather") to the edited image.

Authors modify latent diffusion to condition on image. To do this, they add more input channels to the first convolutional layer (initializing their output weights with zeros). This means that U-Net now accepts channel-wise concatenation of two VAE-encoded images. As usual, the model performs iterative denoising, starting from the latent noise, but now, instead of conditioning on the text description, we condition on both the original image and the editing instruction.

To gather training data, authors start with 700 human-made text triplets (original description, editing instruction, result description). They fine-tune GPT-3 Davinci on this dataset and then apply fine-tuned model to various image descriptions from the LAION-Aestetics V2 6.5+, to collect. As a result, authors obtain hundreds of thousands of editing triplets. Then, authors generate images from both input description and edited description, using Stable Diffusion with Prompt2Prompt technique (fig. 3), using different values of Prompt2Prompt hyperparameter "p" (the fraction of denoising steps p with shared attention weights).

Author drop input and edited descriptions and retain triplets (input image, editing insctruction, edited image). Further, these data aree filtered by aestetics score and aestetics score difference between image pairs, and also with CLIP-based directional similariy metric. In this way, the authors obtain a large (450K samples) model-generated training dataset.

Authors train the above described model on this data. Authors introduce 2 parameters which control classifier-free guidance scales over image conditioning and instruction conditioning. They can be adjusted to trade off how strongly the generated samples correspond with the input image and how strongly they correspond with the edit instruction (fig 4).

The paper contains a lot of editing examples, as well as failure cases (fig. 13, 14). For example, model is not capable of performing viewpoint changes, can make undesired excessive changes to the image, can sometimes fail to isolate the specified object, and has difficulty reorganizing or swapping objects with each other. It struggles with counting numbers of objects and with spatial reasoning (e.g., “move it to the left of the image”, “swap their positions”, or “put two cups on the table and one on the chair”), just as in Stable Diffusion and Prompt2Prompt.

An important peculiarity of this work is that almost all data were collected using another models.

# CV: pretraining

**Kornblith, S., Shlens, J., & Le, Q. V. (2018). Do Better ImageNet Models Transfer Better? arXiv, 1805.08974. Retrieved from https://arxiv.org/abs/1805.08974v3**

Downstream classification performance generally correlates with imagenet performance, however on two small fine-grained image classification datasets, pretraining on ImageNet provides minimal benefits. ImageNet architectures generalize well across datasets, but ImageNet features are less general than previously suggested.

# SSM, World models with CV, NLP

**Wong, L., Grand, G., Lew, A. K., Goodman, N. D., Mansinghka, V. K., Andreas, J., & Tenenbaum, J. B. (2023). From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought. arXiv, 2306.12672. Retrieved from https://arxiv.org/abs/2306.12672v2**

A context-sensitive mapping from natural language into a probabilistic language of thought (PLoT)--a general-purpose symbolic substrate for generative world modeling.

We model thinking with probabilistic programs, an expressive representation for commonsense reasoning; and we model meaning construction with large language models (LLMs), which support broad-coverage translation from natural language utterances to code expressions in a probabilistic programming language.

We show that LLMs can generate context-sensitive translations that capture pragmatically-appropriate linguistic meanings, while Bayesian inference with the generated programs supports coherent and robust commonsense reasoning.

**Lin, Jessy, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel, Dan Klein, and Anca Dragan. 2023. “[Dynalang] Learning to Model the World with Language.” arXiv [cs.CL]. arXiv. http://arxiv.org/abs/2308.01399.**

The authors propose an RL agent named Dynalang. It is a modification of DreamerV3 with additional text inputs. This means that each observation o_t is a combination of one video frame and one text token (one token per each frame!). The world model is able to predict video, reward and text in the future (fig. 1). By design, Dynalang can be pretrained on text-only or video-only data without actions or task reward.

"While previous settings specify that language such as instructions arrive at the beginning of an episode, we are interested in enabling agents to act in more flexible settings where they face a continuous stream of video and text, as in the real world."

This is motivated in the following way. If the environment contains language signals, then language is not always directly connected to instructions, but may describe a world state or knowledge about how the world works (for example, "the top left button turns off the TV" or "I already vacuumed the living room"). So, authors include language into world modeling.

Authors evaluate Dynalang on tasks with virtual environments, for example on vision-language navigation (fig. 7). Also authors evaluate on Messenger task, which tests whether agents can read text manuals describing game dynamics to achieve high scores. In this task, pretraining on TinyStories (a dataset of 2M short stories generated by GPT-3.5 and GPT-4) increase performance. Authors do not evaluate language modeling performance of Dynalang, but provide sampled 10-token generations conditioned on a prefix of 50 tokens for validation examples in TinyStories (Appendix E).

**Hafner, Danijar, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. 2023. “[DreamerV3] Mastering Diverse Domains through World Models.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/2301.04104.**

The authors propose an algorithm DreamerV3 that is applicable to a wide range of diverse tasks (robot locomotion, manipulation, Minecraft etc.) out of the box - without hyperparemeters tuning. Authors evaluate DreamerV3 on 7 domains that include that continuous and discrete actions, visual and low-dimensional inputs, dense and sparse rewards, different reward scales, 2D and 3D worlds, and procedural generation.

DreamerV3 achieves strong performance on all domains and demonstrate favorable scaling properties: increasing the model size monotonically improves both final performance and data-efficiency. DreamerV3 is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula, solving a long-standing challenge in artificial intelligence.

DreamerV3 consists of 3 neural networks: the world model based on the recurrent state space model (RSSM), the critic and the actor (fig. 3). In RSSM a hidden state consists of deterministic and stochastic parts. Like in the DreamerV2, the stochastic state is a vector of multiple categorical variables (see fig. 2 in DreamerV2 paper). Given a hidden state, the actor estimates the distribution in action space to maximize the expected reward. Given a hidden state, the critic learns to predict the expected reward under the current actor behavior.

IMO next. While i'm not a specialist in RL, it seems like the goal of the actor is to produce actions with a cheap network. Instead of using actor, we could use computational-heavy inference-time lookahead planning, as done in PlaNet. The goal of the critic is to help the actor learn: it predicts the estimated reward beyond the prediction horizon of 16 steps to provide a target for training the actor.

Autors carefully tune the algorithm. Some but not all details:

1. Observations are preprocessed by the symlog function (fig. 4), and the restoration MSE loss is applied to the preprocessed observation (eq. 1). The same goes for reward. This allows the optimization process to quickly move the network predictions to large values when needed.
2. The KL-divergence loss in RSSM  is split into two terms by applying the stop-gradient to the first and to the second operand (eq. 5). These terms are weighted with different weights. So, L_dyn and L_rep is actually the same loss, but backpropagated through different submodules.
3. The reward is predicted using twohot encoding to better capture distributions with multiple modes (eq. 9, 10). 
4. Because the critic regresses targets that depend on its own predictions, authors stabilize learning by regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters.

The example predictions of the DreamerV3 world model are shown at fig. 5. From 5 input frames, the model predicts 45 frames into the future given the action sequence.

**Hafner, Danijar, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. 2018. “[PlaNet, RSSM] Learning Latent Dynamics for Planning from Pixels.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1811.04551.**

The authors propose RSSM (Recurrent State Space Model) and use it to build PlaNet (Deep Planning Network), a model-based agent that is able to learn from videos and actions.

(In RL, a model-based agent is an agent that learns a model of the world. If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm. If it can't, then it’s a model-free algorithm.)

Authors argue that a model-based approach offer several benefits over model-free RL. Firstly, it can be more data-efficient, because world modeling usually require predicting in pixel space, and loss in pixel spae provides rich training signal. Secondly, learned world model can be independent of any specific task, and thus have more potential to transfer to other tasks in the same environment. (read even more reasons in the paper, sec. 1)

To to enable fast planning, authors propose to learn the world dynamics in a compact latent space, instead of directly predicting pixels in the future moments. The authors start with formulating a model with discrete time, hidden states (latents) s_t, image observations o_t, actions a_t and scalar rewards r_t, that follow the stochastic dynamics with a fixed initial state s_0:

Transition function: s_t ~ p(s_t | s_{t-1}, a_{t-1})
Observation function: o_t ~ p(o_t | s_t)
Reward function: r_t ~ p(r_t | s_t)
Action policy: a_t ~ p(a_t | o_{≤t}, a_{≤t})

Transition function, observation function and reward function are learnable, parametrized by neural networks (deconvolutional network for observation function). Action policy is not learned. Instead, given given a world and reward models, optimal actions can be obtained by inference-time optimization: searching for action sequence that maximizes summary reward.

To train the above model by MLE (maximum likelihood estimation), we should search for the combination of unknown values (hidden states and weights of neural nets) that maximize the log-probability of the dataset, that is sum of log-probabilities for all samples. To achieve this, we could learn hidden states for all samples by gradient descent. But if we go this way, how do we obtain hidden states on inference time from observations? Given an observation and transition function, obtaining posterior probabilities for hidden states is intractable.

The same problem arose in autoencoders and were solved by training encoder that approximate posterior for hidden states. The authors go the same way and train convolutional encoder q(s_t | s_{t-1}, a_{t-1}, o_t) to infer approximate posteriors for hidden states.

So, in total, we get 4 trainable functions: transition function, observation function, reward function and encoder. Interestingly, the observation function is not used for planning, so may be dropped after training.

To train all the functions, authors use variational lower bound of the data log-probability (eq. 3). This loss consists of two terms, when expectations can be estimated by a single reparemetrized sample. The first term is a reconstruction error: if we sequentially apply encoder, sampling and observation function, how well do we recover the original input? The second term is a KL-divergence between the hidden state distributions obtained from encoder and from transition function: they should match to minimize this term.

Importantly, the transition function is stochastic: it returns distribution parameters that we sample from to obtain the next state. Such models are called state space models (SMM) and are different from RNN, where the transition function is deterministic. If in SSM we reduce the predicted variance to zero, we end up with RNN. The authors argue that the stochastity is very important, since it allows the model to capture different possible futures. But SSM have a problem to pass some information through many hidden states, since multiple sampling operations may distort information. So, authors propose to combine SSM and RNN giving RSSM, where the hidden state in each moment consists of the deterministic and stochastic parts: firstly we predict the deterministic part from the previous time moment, and then predict the stochastic part (fig. 2).

The authors point out another shortcoming of the RSSM. If observations at every time moments are accessible, this makes learning the transition function less important. In eq. 3 the loss even does not propagate through multiple time steps. To cure this, authors propose latent overshooting: a technique when we predict the future hidden states through N time steps (from t to t+N) without accessing the intermediate observations, and then minimize KL-divergence between the predicted state distribution and the distribution obtained from the encoder at time t+N (fig. 3c, loss at eq. 6, 7). An alternative to this is the earlier proposed observation overshooting: predicting pixels through N time steps without accessing the intermediate observations (fig. 3b) - but this is too expensive, since we should make a pass through observation model too many times.

Based on RSSM, authors propose PlaNet algorithm (Algorithm 1). Starting from some episodes, authors fit RSSM on them, and then do actions using fitted model to collect new training data. This approach require acting and running the true world model (game algorithm), that is possible in virtual environments.

In the experimental parts, authors train the model to solve multiple image-based continuous control tasks of the DeepMind control suite. In all tasks, the only observations are third-person camera images of size 64x64x3 pixels. The authors conclude that both deterministic and stochastic parts are important. The stochastic component is even more important – the agent does not learn without it.

IMO thoughts next.

It's interesting to think about the taxonomy of "world model". This is the model of how some environment of interest works. So it requires a transition function between world states S, when the states may contain fixed and stochastic (this means not fully known) parts. To train the model we use sequences of observations O (videos, for example). There may be 4 cases.

1. We use observations as world states: S = O. In this case we come to autoregressive video modeling.
2. We train only a function O -> S. If we train such model end-to-end, the model may collapse to the solution when world states are noninformative, for example are constantly zero, and are trivially predicted. In general, the prediction loss will encourage O -> S function to extract features that are most easily predicted.
3. We train only a function S -> O. In this case, how do we obtain S given O? Exact algorithm may be intractable, and approximate algorithm may require inference-time optimization, that is costly.
4. We train both functions O -> S and S -> O. In this case we come to the autoencoder with time axis, as done in SSM and in this work.

There is one possible reason why splitting the internal state into determinstic and stochastic parts is beneficial. Each observation may not show us the full world, while observation history may allow the model to capture the full world. In this case, h_t contains information about the full world, and s_t ~ p(s_t | h_t) contains only currently visible information. So, calculating KL-divergence between s_t and encoded observation seems more suitable. RSSM also looks similar to the world model from Ha and Schmidhuber, 2018 that also contains deterministic h_t and stochastic z_t.

While transition in SSM requires sampling, this is quite similar to dropout. Dropout can be also viewed as sampling from distribution. So, it turns out that RNN with inference-time dropout is a special case of SSM?

When pixels are observations, the problem in RSSM is the pixel-wise loss. Some aspects of the environment may be irrelevant and better should not be stored in the hidden state, while other aspects (for example, the shape of the key) may be only slightly visible in pixel space, so they do not contribute much to the pixel-wise loss.

The second problem here is a general RNN problem: how do we compress the world state, that may be very complex, into a vector of the fixed size? A common sense tells that the more is the size of uncompressed information, the more is the size of compressed information. A hidden state is the compressed information about the agent's experience. From this side, transformer architecture seems more suitable, since in transformer the agent's memory grows with the growth of the context. The optimal way seems to compress information nearly logarithmically, when compressed experience size should be proportional to the logarithm of uncompressed experience size. However, this is not the case both in transformers and in RNN/SSM.

The notion of internal (hidden) state seems also ambiguous in some tasks. Should an internal state contain information about precise pixel boundaries? If yes, it should be very large. If no, how will the agent shoot enemies, if the task require precise shooting?

**Ha, David, and Jürgen Schmidhuber. 2018. “World Models.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1803.10122.**

Authors propose a model-based reinforcement learning agent that successfully solves car racing and VisDoom tasks with visual input in the OpenAI Gyn environment (fig. 9, 14).

If the network that learns action policy is large, with high representational capacity, then it will probably suffer from the credit assignment problem. Hence in practice, smaller networks are used as they iterate faster to a good policy during training. However, to unleash the power of deep learning and to model complex environments we want a large neural network. To solve this dilemma, authors train a large task-independent world model and a small, task-dependent controller network on top of it. A small controller lets the training algorithm focus on the credit assignment problem on a small search space, while not sacrificing capacity and expressiveness via the larger world model.

Authors use a probabilistic generative world model that includes several components. Firstly, a variational autoencoder (VAE) compresses each video frame to a small latent vector z_t. Authors train VAE on a given set of frames and then freeze it. Secondly, authors use RNN with hidden state h_t: each time step in accepts latent vector z_{t-1}, action a_{t-1} and previous hidden state h_{t-1} and returns new hidden state h_t and a probability distribution p(z_t) in form of gaussian mixture. Such RNN that outputs gaussian misture are called Mixture Density Networks (MDN-RNN) and were previously used in several works. From Graves, 2013: "A subset of the outputs are used to define the mixture weights, while the remaining outputs are used to parameterise the individual mixture components. The mixture weight outputs are normalised with a softmax function to ensure they form a valid discrete distribution, and the other outputs are passed through suitable functions to keep their values within meaningful range (for example the exponential function is typically applied to outputs used as scale parameters, which must be positive)."

On top of the world model authors train a controller that accepts z_t and h_t and linearly projects them to the action a_t (eq. 1). A little number of parameters of controller allows to explore more unconventional ways to train it – for example, even using evolution strategies.

The overall training procedure (sec. 3.2) is the following:

1. Collect 10K rollouts from a random policy (that is, what happened if we act randomly). As a result, we obtain 10K frame sequences.
2. Train VAE on all the frames.
3. Train MDN-RNN to model P(z_t) (probably using KL-loss).
4. Train controller with CMA-ES method (evolution strategy) to maximize the expected cumulative reward.

Interestingly, a controller may be trained using only the learned world model, that is, without the game engine. This may be especially useful for the agent acting in the read world. Authors add extra uncertainty into such virtual environment, thus making the training more challenging. They do this by increasing the temperature parameter during the sampling process of z_t. Agents that perform well in higher temperature settings generally perform better in the normal setting. Increasing temperature helps prevent controller from taking advantage of the imperfections of the learned world model.

VAE is trained independently of RNN world model, and so it may encode parts of the observations that are not relevant to a task. By training together with a world model that predicts rewards, the VAE may learn to focus on task-relevant areas of the image, but the tradeoff here is that we may not be able to reuse the VAE effectively for new tasks without retraining.

Using a gaussian mixture in RNN allows to capture multiple different futures with random discrete events, such as whether a monster decides to shoot a fireball or stay put. "For instance, if we set the temperature parameter to a very low value of 0.1, ... the monsters inside this dream environment fail to shoot fireballs, no matter what the agent does, due to mode collaps. The M model is not able to jump to another mode in the mixture of Gaussian model where fireballs are formed and shot. Whatever policy learned inside of this dream will achieve a perfect score of 2100 most of the time, but will obviously fail when unleashed into the harsh reality of the actual world, underperforming even a random policy."

# NLP: interpreting, robustness, uncertainty, shortcuts

**Kalai, A. T., & Vempala, S. S. (2023). Calibrated Language Models Must Hallucinate. arXiv, 2311.14648. Retrieved from https://arxiv.org/abs/2311.14648v1**

The authors aim to prove that statistically calibrated generative model of some text corpus should hallucinate rare facts. Howeverr, authors' formalization is rather limited, since they consider only promptless generation of facts (with no previous context). Also, authors consider their own (computationally intractable) notion of calibration, different from standard token-based ones.

In this simplified setting, the authors prove that any calibrated model, regardless of its architecture, will hallucinate on arbitrary facts. This means that in unconditional generation unexisting facts may appear, and the probability of generating a hallucination is close to the fraction of facts that occur exactly once in the training data.

"Arbitrary facts" means that the truthfulness of the facts cannot be valiated by rules (like 572 < 120523 can be valiated by math rules). Examples of these facts are "5W facts" (Who-What-When-Where-Why), for example "Alexa Wilkins had a tuna sandwich at Salumeria for lunch last Tuesday".

The authors note that they only study one statistical source of hallucination, while there are many other types of hallucination and reasons LMs may hallucinate beyond pure statistics.