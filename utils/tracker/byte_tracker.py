import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from utils.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        np.float = float
        np.int = int
        np.object = object
        np.bool = bool

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.   #以边界框格式获取当前位置`(左上x,左上y,宽度，高度）
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:5].copy()   #改
        # ret[2] *= ret[3]   #  改
        # ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        # ret[2:] += ret[:2]   #不将xywh改回tlbr
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        # ret[:2] += ret[2:] / 2      #不将xywh改回tlbr
        # ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):    # gai--lm
        ret = np.asarray(tlbr).copy()
        # ret[2:] -= ret[:2]   #本来就是xywhθ格式
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# class BYTETracker(object):
#     def __init__(self, args, frame_rate=30):
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]
#
#         self.frame_id = 0
#         self.args = args
#         #self.det_thresh = args.track_thresh
#         self.det_thresh = args.track_thresh + 0.1
#         self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
#         self.max_time_lost = self.buffer_size
#         self.kalman_filter = KalmanFilter()
#
#     def update(self, output_results, img_info, img_size):
#         self.frame_id += 1
#         activated_starcks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []
#
#         if output_results.shape[1] == 5:   #如果维度为5
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         elif output_results.shape[1] == 7:   #如果维度为7
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 5]
#             bboxes = output_results[:, :5]  # x1y1x2y2            #gai--lm
#             num_classlm = output_results[:, 6]   # 加--lm
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 5] * output_results[:, 6]   #gai--lm
#             bboxes = output_results[:, :5]  # x1y1x2y2            #gai--lm
#             num_classlm = output_results[:, 7]   # 加--lm
#         img_h, img_w = img_info[0], img_info[1]
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))    #缩放比
#         bboxes /= scale
#
#         remain_inds = scores > self.args.track_thresh
#         inds_low = scores > 0.1
#         inds_high = scores < self.args.track_thresh
#
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = bboxes[inds_second]
#         dets = bboxes[remain_inds]
#         scores_keep = scores[remain_inds]
#         scores_second = scores[inds_second]
#
#         if len(dets) > 0:
#             '''Detections'''   #检测
#             detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
#                           (tlbr, s) in zip(dets, scores_keep)]
#         else:
#             detections = []
#
#         ''' Add newly detected tracklets to tracked_stracks'''   #将新检测到的 tracklet 添加到 tracked_stracks
#         unconfirmed = []
#         tracked_stracks = []  # type: list[STrack]
#         for track in self.tracked_stracks:    #检测时未进
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)
#         #detection[]存放的当前帧中的高分检测框
#         #unconfirmed[]存放上一帧中的新的tracker
#         #tracked_stracks[]存放上一帧中正常的激活状态的tracker
#
#
#         #第 2 步：第一次关联，带有高分检测框
#         ''' Step 2: First association, with high score detection boxes'''
#         '''
#         这里关联的（上一帧）trackers（状态是激活的），正常的，丢失的都可与高分检测框进行匹配，注意这里是没有新的，删除的轨迹进行匹配
#         若匹配到正常的tracker，则更新位置，放入activate_stracks 列表
#         若匹配到丢失的tracker，则将种类从丢失改为正常，状态改为激活，放入 refind_stracks 列表
#         '''
#         # 将正常的 tracker 和 已丢失的 tracker，根据 tracker_id 放在一起
#         strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  #第一次[]  第二次 [OT_1_(1-1), OT_2_(1-1)]
#
#         #    用 KF 预测当前位置
#         # Predict the current location with KF,更新均值和方差
#         # 卡尔曼预测，例 len(strack_pool)=17
#         STrack.multi_predict(strack_pool)   #运用 运算
#
#         # 计算trackers和当前(高分)检测框的iou，然后 dists=1-iou，即iou越大dsits中的值越小，代价矩阵。例len(detections)=20，dists.shape=[17, 20]
#         dists = matching.iou_distance(strack_pool, detections)
#         if not self.args.mot20:
#             dists = matching.fuse_score(dists, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
#
#         # 根据匹配到的检测框，更新参数
#         for itracked, idet in matches:   #match是一个形如[(a,b),(),(),(),(),(),()]的列表,其中itracke代表遍历了同a一列的元素,idet代表遍历了同b一列的元素
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 # 正常的（state=1） trackers 直接使用 det 更新
#                 # 更新 tracklet_len，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
#                 track.update(detections[idet], self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 # 若不是正常tracker（这里只有丢失的），丢失的tracker根据det更新参数
#                 # tracklet_len=0，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)          # 重新放入列表
#
#
#
#         #第 3 步：第二次关联，使用低分检测框
#         ''' Step 3: Second association, with low score detection boxes'''
#         # association the untrack to the low score detections
#         if len(dets_second) > 0:
#             '''Detections'''
#             detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
#                           (tlbr, s) in zip(dets_second, scores_second)]
#         else:
#             detections_second = []
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)
#
#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if not track.state == TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)
#
#
#         #       处理未经确认的轨道，通常只有一个开始帧的轨道
#         '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
#         detections = [detections[i] for i in u_detection]
#         dists = matching.iou_distance(unconfirmed, detections)
#         if not self.args.mot20:
#             dists = matching.fuse_score(dists, detections)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections[idet], self.frame_id)
#             activated_starcks.append(unconfirmed[itracked])
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)
#
#         # 第 4 步：初始化新的 stracks
#         """ Step 4: Init new stracks"""
#         for inew in u_detection:
#             track = detections[inew]
#             if track.score < self.det_thresh:
#                 continue
#             track.activate(self.kalman_filter, self.frame_id)
#             activated_starcks.append(track)
#
#         #  更新状态
#         """ Step 5: Update state"""
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)
#
#         # print('Ramained match {} s'.format(t4-t3))
#
#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         # get scores of lost tracks
#         output_stracks = [track for track in self.tracked_stracks if track.is_activated]
#
#         return output_stracks



#修改的bytetrack
##修改的bytetrack

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:   #如果维度为5
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        elif output_results.shape[1] == 7:   #如果维度为7
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 5]
            bboxes = output_results[:, :5]  # x1y1x2y2            #gai--lm
            num_classlm = output_results[:, 6]   # 加--lm
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 5] * output_results[:, 6]   #gai--lm
            bboxes = output_results[:, :5]  # x1y1x2y2            #gai--lm
            num_classlm = output_results[:, 7]   # 加--lm
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))    #缩放比
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''   #检测
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for   #STrack.tlbr_to_tlwh是使用上面的tlbr_to_tlwh函数
                          (tlbr, s) in zip(dets, scores_keep)]  #组合   这行代码在列表推导式中为每一个 (tlbr, s) 元组创建一个 STrack 实例，参数为 tlwh 和 s。
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''   #将新检测到的 tracklet 添加到 tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]   #定义了一个空列表 tracked_stracks，类型为 list[STrack]
        for track in self.tracked_stracks:    #检测时未进
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        #detection[]存放的当前帧中的高分检测框
        #unconfirmed[]存放上一帧中的新的tracker
        #tracked_stracks[]存放上一帧中正常的激活状态的tracker


        #第 2 步：第一次关联，带有高分检测框
        ''' Step 2: First association, with high score detection boxes'''
        '''
        这里关联的（上一帧）trackers（状态是激活的），正常的，丢失的都可与高分检测框进行匹配，注意这里是没有新的，删除的轨迹进行匹配
        若匹配到正常的tracker，则更新位置，放入activate_stracks 列表
        若匹配到丢失的tracker，则将种类从丢失改为正常，状态改为激活，放入 refind_stracks 列表
        '''
        # 将正常的 tracker 和 已丢失的 tracker，根据 tracker_id 放在一起
        # strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  #第一次[]  第二次 [OT_1_(1-1), OT_2_(1-1)]        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  #第一次[]  第二次 [OT_1_(1-1), OT_2_(1-1)]
        strack_pool = tracked_stracks #第一次[]  第二次 [OT_1_(1-1), OT_2_(1-1)]


        #    用 KF 预测当前位置
        # Predict the current location with KF,更新均值和方差
        # 卡尔曼预测，例 len(strack_pool)=17
        STrack.multi_predict(strack_pool)   #运用 运算

        # 计算trackers和当前(高分)检测框的iou，然后 dists=1-iou，即iou越大dsits中的值越小，代价矩阵。例len(detections)=20，dists.shape=[17, 20]
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:  #
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 根据匹配到的检测框，更新参数
        for itracked, idet in matches:   #match是一个形如[(a,b),(),(),(),(),(),()]的列表,其中itracke代表遍历了同a一列的元素,idet代表遍历了同b一列的元素
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 正常的（state=1） trackers 直接使用 det 更新
                # 更新 tracklet_len，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 若不是正常tracker（这里只有丢失的），丢失的tracker根据det更新参数
                # tracklet_len=0，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)          # 重新放入列表



        #第 3 步：第二次关联，使用低分检测框
        # ''' Step 3: Second association, with low score detection boxes'''
        # # association the untrack to the low score detections
        # if len(dets_second) > 0:
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
        #                   (tlbr, s) in zip(dets_second, scores_second)]
        # else:
        #     detections_second = []
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # for itracked, idet in matches:
        #     track = r_tracked_stracks[itracked]
        #     det = detections_second[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)
        #
        # for it in u_track:
        #     track = r_tracked_stracks[it]
        #     if not track.state == TrackState.Lost:
        #         track.mark_lost()
        #         lost_stracks.append(track)


        #       处理未经确认的轨道，通常只有一个开始帧的轨道
        # '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # detections = [detections[i] for i in u_detection]
        # dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # for itracked, idet in matches:
        #     unconfirmed[itracked].update(detections[idet], self.frame_id)
        #     activated_starcks.append(unconfirmed[itracked])
        # for it in u_unconfirmed:
        #     track = unconfirmed[it]
        #     track.mark_removed()
        #     removed_stracks.append(track)

        # 第 4 步：初始化新的 stracks
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        #  更新状态
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
