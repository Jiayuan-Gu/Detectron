import numpy as np
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python import gradient_checker
from detectron.ops.map_grid_generator import MapGridGeneratorOp


def GridEmbedding(net, blob_in, emb_dim, wave_length=1000.):
    num_coords = 2
    dim = emb_dim / (2 * num_coords)
    range_vec = np.arange(dim, dtype=np.float32) / float(dim)
    base_vec = np.ones_like(range_vec) * wave_length
    range_vec = np.power(base_vec, range_vec)
    range_vec = net.Const(range_vec, blob_out='range_vec')
    blobs_in = net.Split([blob_in], [blob_in + '_split' + str(idx) for idx in range(num_coords)], axis=1)
    blobs_out = []
    for idx, blob_in in enumerate(blobs_in):
        tiled_blob_in = net.Tile([blob_in], blob_in + '_tiled', axis=1, tiles=dim)
        div_mat = net.Div([tiled_blob_in, range_vec], '_grid_emb_div_mat' + str(idx), broadcast=1, axis=1)
        sin_mat = net.Sin(div_mat, '_grid_emb_sin_mat' + str(idx))
        cos_mat = net.Cos(div_mat, '_grid_emb_cos_mat' + str(idx))
        blobs_out += [sin_mat, cos_mat]

    embedding, _ = net.Concat(
        blobs_out, ['grid_embedding', '_grid_emb_concat'],
        axis=1
    )
    return embedding


def main0():
    net = core.Net('playground')
    feat_map = net.ConstantFill([], ['feat_map'], shape=(1, 2, 6, 10), value=1.)
    map_grid = net.Python(
        MapGridGeneratorOp(16).forward
    )(feat_map, 'map_grid')
    embedding = GridEmbedding(net, map_grid, 16)
    # print(net.Proto())

    workspace.CreateNet(net)
    workspace.RunNet(net.Proto().name)
    print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
    for name in workspace.Blobs():
        blob = workspace.FetchBlob(name)
        # print("{}:\n{}, {}".format(name, blob, blob.shape))
        print("{}:{}\n".format(name, blob.shape))


def main1():
    import detectron.utils.c2 as c2_utils
    c2_utils.import_custom_ops()
    assert 'AttentionCropGrid' in workspace.RegisteredOperators()

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
    # with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
        op = core.CreateOperator(
            'AttentionCropGrid', ['grid_feat', 'rois', 'saliency_map'],
            ['crop_grid_feat'],
            spatial_scale=1.0, max_area=196, num_coords=2,
        )

        # feat_map = np.ones((1, 2, 4, 4), dtype=np.float32)
        grid_feat = np.ones((1, 1 * 2, 2 * 4 - 1, 2 * 4 - 1), dtype=np.float32)
        rois = np.array([[0, 0, 0, 3, 3], [0, 1, 1, 2, 2]], dtype=np.float32)
        # rois = np.array([[0, 0, 0, 3, 3]], dtype=np.float32)
        # saliency_map = np.zeros((1, 1, 4, 4), dtype=np.float32)
        saliency_map = np.ones((1, 1, 4, 4), dtype=np.float32)

        for idx, arr in enumerate([grid_feat, rois, saliency_map]):
            workspace.FeedBlob(op.input[idx], arr,
                               device_option=core.DeviceOption(caffe2_pb2.CUDA, 0))
        workspace.RunOperatorOnce(op)
        crop_grid_feat = workspace.FetchBlob('crop_grid_feat')
        print(crop_grid_feat)

        gc = gradient_checker.GradientChecker(
            stepsize=0.001,
            threshold=0.001,
            # device_option=core.DeviceOption(caffe2_pb2.CPU, 0),
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0),
        )

        # res, grad, grad_estimated = gc.CheckSimple(
        #     op, [grid_feat, rois, saliency_map], 0, [0]
        # )
        res, grad, grad_estimated = gc.CheckSimple(
            op, [grid_feat, rois, saliency_map], 2, [0]
        )
        print(grad)
        print(grad_estimated)

        # assert grad.shape == grad_estimated.shape, 'Fail check: grad.shape != grad_estimated.shape'


def main2():
    import detectron.utils.c2 as c2_utils
    c2_utils.import_custom_ops()
    assert 'AttentionAggregation' in workspace.RegisteredOperators()

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
    # with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
        op = core.CreateOperator(
            'AttentionAggregation', ['feat_map', 'rois', 'weight'],
            ['att_feat'],
            spatial_scale=1.0, max_area=32, num_coords=2,
        )

        feat_map = np.ones((1, 3, 4, 4), dtype=np.float32)
        rois = np.array([[0, 0, 0, 3, 3], [0, 1, 1, 2, 2]], dtype=np.float32)
        # rois = np.array([[0, 0, 0, 3, 3]], dtype=np.float32)
        weight = np.ones((len(rois), 16, 32), dtype=np.float32)

        for idx, arr in enumerate([feat_map, rois, weight]):
            workspace.FeedBlob(op.input[idx], arr,
                               device_option=core.DeviceOption(caffe2_pb2.CUDA, 0))
                               # device_option=core.DeviceOption(caffe2_pb2.CPU, 0))
        workspace.RunOperatorOnce(op)
        att_feat = workspace.FetchBlob('att_feat')
        print(att_feat)
        print(att_feat.shape)

        gc = gradient_checker.GradientChecker(
            stepsize=0.001,
            threshold=0.001,
            # device_option=core.DeviceOption(caffe2_pb2.CPU, 0),
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0),
        )

        res, grad, grad_estimated = gc.CheckSimple(
            op, [feat_map, rois, weight], 0, [0]
        )
        print(grad)
        # print(grad_estimated)

        # assert grad.shape == grad_estimated.shape, 'Fail check: grad.shape != grad_estimated.shape'


def main3():
    net = core.Net('playground')
    feat_map = net.ConstantFill([], ['feat_map'], shape=(1, 2, 7, 7), value=1.)
    slice_map = net.Slice([feat_map], 'slice_map', starts=[0, 0, 0, 1], ends=[-1, -1, -1, -2])
    slice_map2 = net.Slice([slice_map], 'slice_map2', starts=[0, 0, 1, 0], ends=[-1, -1, -2, -1])
    pooled_map = net.AveragePool2D(slice_map2, 'feat_map_pool', kernel=1, stride=2, legacy_pad=1)
    # print(net.Proto())

    workspace.CreateNet(net)
    workspace.RunNet(net.Proto().name)
    print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
    for name in workspace.Blobs():
        blob = workspace.FetchBlob(name)
        # print("{}:\n{}, {}".format(name, blob, blob.shape))
        print("{}:{}\n".format(name, blob.shape))


def main4():
    import detectron.utils.c2 as c2_utils
    c2_utils.import_custom_ops()
    assert 'SoftmaxV1' in workspace.RegisteredOperators()

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        op = core.CreateOperator(
            'SoftmaxV1', ['X'],
            ['Y'],
        )

        # X = np.ones((4, 4), dtype=np.float32)
        X = np.random.normal(size=(10, 16, 32)).astype(np.float32)

        for idx, arr in enumerate([X]):
            workspace.FeedBlob(op.input[idx], arr,
                               device_option=core.DeviceOption(caffe2_pb2.CUDA, 0))
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        print(Y)
        print(Y.sum(axis=-1))

        gc = gradient_checker.GradientChecker(
            stepsize=0.001,
            threshold=0.001,
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0),
        )

        res, grad, grad_estimated = gc.CheckSimple(
            op, [X], 0, [0]
        )
        print(grad)
        print(grad_estimated)


if __name__ == '__main__':
    main4()