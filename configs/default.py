dataset = dict(
    name='cityscapes',
    cross_validation_split=0,
)
loss = dict(
    use_joint_edgeseg_loss=True,
    use_img_wt_loss=True,
    edge_weight=1.0,
    seg_weight=1.0,
    att_weight=1.0,
    dual_weight=1.0,
)
