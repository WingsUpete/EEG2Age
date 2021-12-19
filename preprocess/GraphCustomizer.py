import os
import sys
import argparse

import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

from GraphBuilder import pushGraphEdge, matOD2G


GRAPH_STRUCTURE_PATH_DEFAULT = 'testGraph.txt'
OUT_DIR_DEFAULT = './'
OUT_FN_DEFAULT = 'graph_customize.dgl'


def identifyChannels(channel_names):
    """
    Gives each channel an id and return a dictionary
    """
    channels_dict = {}
    for i in range(len(channel_names)):
        channels_dict[channel_names[i]] = i
    return channels_dict


def formFCConnections(channel_ids, mat, sL, dL):
    for sid in channel_ids:
        for did in channel_ids:
            mat[sid][did] = 1
            sL, dL = pushGraphEdge(gSrc=sL, gDst=dL, wList=None, src=sid, dst=did, weight=None)
    return mat, sL, dL


def constructCustomizedGraph(customization_path, out_dir):
    if not os.path.exists(customization_path):
        sys.stderr.write('%s does not exist!\n' % customization_path)
        exit(-1)

    with open(customization_path) as f:
        lines = f.readlines()
        channels = identifyChannels(lines[0].strip().split(','))
        mat = torch.zeros(len(channels), len(channels))
        sL, dL = [], []
        for i in range(len(lines)):
            if i == 0:
                continue
            curRegionChannels = lines[i].strip().split(',')
            curRegionChannelIDs = [channels[channelName] for channelName in curRegionChannels]
            mat, sL, dL = formFCConnections(curRegionChannelIDs, mat, sL, dL)
        g = matOD2G(mat=mat, oList=sL, dList=dL, nGNodes=len(channels))
        outGPath = os.path.join(out_dir, OUT_FN_DEFAULT)
        dgl.save_graphs(outGPath, g)
        print('> Generated graph saved to %s' % outGPath)


if __name__ == '__main__':
    """
        Usage Example:
        python GraphCustomizer.py -g testGraph.txt -o ./
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--graph_structure', type=str, default=GRAPH_STRUCTURE_PATH_DEFAULT, help='Specify the txt file path specifying a graph structure, default = {}'.format(GRAPH_STRUCTURE_PATH_DEFAULT))
    parser.add_argument('-o', '--out', type=str, default=OUT_DIR_DEFAULT, help='Specify the output folder, default = {}'.format(OUT_DIR_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    constructCustomizedGraph(customization_path=FLAGS.graph_structure, out_dir=FLAGS.out)
