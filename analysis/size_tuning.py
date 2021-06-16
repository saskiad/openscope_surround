#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import matplotlib.pyplot as plt


def do_sweep_mean(x):
    return x[30:90].mean()


def do_sweep_mean_shifted(x):
    return x[30:40].mean()


def do_eye(x):
    return x[30:35].mean()


class SizeTuning:
    def __init__(self, expt_path, eye_thresh, cre, area, depth):

        self.expt_path = expt_path
        self.session_id = self.expt_path.split('/')[-1].split('_')[-2]

        self.eye_thresh = eye_thresh
        self.cre = cre
        self.area = area
        self.depth = depth

        self.orivals = range(0, 360, 45)
        self.tfvals = [1.0, 2.0]
        self.sizevals = [30, 52, 67, 79, 120]

        # load dff traces
        f = h5py.File(self.expt_path, 'r')
        self.dff = f['dff_traces'][()]
        f.close()

        # load raw traces
        f = h5py.File(self.expt_path, 'r')
        self.traces = f['raw_traces'][()]
        f.close()

        self.numbercells = self.dff.shape[0]

        # load roi_table
        self.roi = pd.read_hdf(self.expt_path, 'roi_table')

        # get stimulus table for center surround
        self.stim_table = pd.read_hdf(self.expt_path, 'drifting_gratings_size')
        # get spontaneous window
        self.stim_table_spont = self.get_spont_table()

        # load eyetracking
        self.pupil_pos = pd.read_hdf(self.expt_path, 'eye_tracking')

        # run analysis
        (
            self.sweep_response,
            self.mean_sweep_response,
            self.sweep_eye,
            self.mean_sweep_eye,
            self.sweep_p_values,
            self.response,
        ) = self.get_stimulus_response()

        #        self.first, self.second = self.cross_validate_response(n_trials=int(self.response[:,:,:,:,2].min()))
        self.metrics, self.OSI, self.DSI, self.DIR = self.get_metrics()

        # save outputs
        self.save_data()

        # plot traces

    def get_spont_table(self):
        '''finds the window of spotaneous activity during the session'''

        spont_start = np.where(np.ediff1d(self.stim_table.Start) > 8000)[0][0]
        stim_table_spont = pd.DataFrame(columns=('Start', 'End'), index=[0])
        stim_table_spont.Start = self.stim_table.End[spont_start] + 1
        stim_table_spont.End = self.stim_table.Start[spont_start + 1] - 1
        return stim_table_spont

    def get_stimulus_response(self):
        '''calculates the response to each stimulus trial. Calculates the mean response to each stimulus condition.
        Only uses trials when the eye position is within eye_thresh degrees of the mean eye position. Default eye_thresh is 10.

Returns
-------
sweep response: full trial for each trial
mean sweep response: mean response for each trial
sweep_eye: eye position across the full trial
mean_sweep_eye: mean of first three time points of eye position for each trial
response_mean: mean response for each stimulus condition
response_std: std of response to each stimulus condition


        '''
        sweep_response = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=np.array(range(self.numbercells)).astype(str),
        )

        sweep_eye = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=('x_pos_deg', 'y_pos_deg'),
        )

        for index, row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                # uses the global dff trace
                sweep_response[str(nc)][index] = self.dff[
                    nc, int(row.Start) - 30 : int(row.Start) + 90
                ]

                # computes DF/F using the mean of the inter-sweep gray for the Fo
            #                temp = self.traces[nc, int(row.Start)-30:int(row.Start)+90]
            #                sweep_response[str(nc)][index] = ((temp/np.mean(temp[:30]))-1)
            sweep_eye.x_pos_deg[index] = self.pupil_pos.x_pos_deg[
                int(row.Start) - 30 : int(row.Start + 90)
            ].values
            sweep_eye.y_pos_deg[index] = self.pupil_pos.y_pos_deg[
                int(row.Start) - 30 : int(row.Start + 90)
            ].values

        mean_sweep_response = sweep_response.applymap(do_sweep_mean)
        mean_sweep_eye = sweep_eye.applymap(do_eye)
        mean_sweep_eye['total'] = np.sqrt(
            ((mean_sweep_eye.x_pos_deg - mean_sweep_eye.x_pos_deg.mean()) ** 2)
            + (
                (mean_sweep_eye.y_pos_deg - mean_sweep_eye.y_pos_deg.mean())
                ** 2
            )
        )

        # make spontaneous p_values
        shuffled_responses = np.empty((self.numbercells, 10000, 60))
        #        idx = np.random.choice(range(self.stim_table_spont.Start, self.stim_table_spont.End), 10000)
        idx = np.random.choice(
            range(
                int(self.stim_table_spont.Start), int(self.stim_table_spont.End)
            ),
            10000,
        )
        for i in range(60):
            shuffled_responses[:, :, i] = self.dff[:, idx + i]
        shuffled_mean = shuffled_responses.mean(axis=2)
        sweep_p_values = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=np.array(range(self.numbercells)).astype(str),
        )
        for nc in range(self.numbercells):
            subset = mean_sweep_response[str(nc)].values
            null_dist_mat = np.tile(shuffled_mean[nc, :], reps=(len(subset), 1))
            actual_is_less = subset.reshape(len(subset), 1) <= null_dist_mat
            p_values = np.mean(actual_is_less, axis=1)
            sweep_p_values[str(nc)] = p_values

        # compute mean response across trials, only use trials within eye_thresh of mean eye position
        response = np.empty(
            (8, 2, 6, self.numbercells, 4)
        )  # ori X TF x size X cells X mean, std, #trials, % significant trials
        response[:] = np.NaN

        for oi, ori in enumerate(self.orivals):
            for ti, tf in enumerate(self.tfvals):
                for si, size in enumerate(self.sizevals):
                    subset = mean_sweep_response[
                        (self.stim_table.Ori == ori)
                        & (self.stim_table.TF == tf)
                        & (self.stim_table.Size == size)
                        & (mean_sweep_eye.total < self.eye_thresh)
                    ]
                    subset_p = sweep_p_values[
                        (self.stim_table.Ori == ori)
                        & (self.stim_table.TF == tf)
                        & (self.stim_table.Size == size)
                        & (mean_sweep_eye.total < self.eye_thresh)
                    ]

                    response[oi, ti, si + 1, :, 0] = subset.mean(axis=0)
                    response[oi, ti, si + 1, :, 1] = subset.std(axis=0)
                    response[oi, ti, si + 1, :, 2] = len(subset)
                    response[oi, ti, si + 1, :, 3] = subset_p[
                        subset_p < 0.05
                    ].count().values / float(len(subset))

        subset = mean_sweep_response[np.isnan(self.stim_table.Ori)]
        subset_p = sweep_p_values[np.isnan(self.stim_table.Ori)]
        response[0, 0, 0, :, 0] = subset.mean(axis=0)
        response[0, 0, 0, :, 1] = subset.std(axis=0)
        response[0, 0, 0, :, 2] = len(subset)
        response[0, 0, 0, :, 3] = subset_p[
            subset_p < 0.05
        ].count().values / float(len(subset))

        return (
            sweep_response,
            mean_sweep_response,
            sweep_eye,
            mean_sweep_eye,
            sweep_p_values,
            response,
        )

    def cross_validate_response(self, n_iter=50, n_trials=15):
        '''Splits the responses into two arrays to enable cross-validation of metrics across multiple iterations

    Parameters
    ----------
    n_iter: number of iterations. Default is 50
    n_trials: total number of trials being used

    Returns
    ------
    response_first: one half of the response array. Shape is 8 X 4 X number cells X number iterations
    response_second: second half of the response array - same shape.
        '''
        cell_trials = np.empty((8, 2, 6, n_trials, self.numbercells))
        cell_trials[:] = np.NaN

        response_first = np.empty((8, 2, 6, self.numbercells, n_iter))
        response_second = np.empty((8, 2, 6, self.numbercells, n_iter))
        response_first[:] = np.nan
        response_second[:] = np.NaN

        #        for si, size in enumerate(self.sizevals):
        #            if size==0:
        cell_trials[0, 0, 0, :, :] = self.mean_sweep_response[
            np.isnan(self.stim_table.Ori)
            & (self.mean_sweep_eye.total < self.eye_thresh)
        ].values[:n_trials]
        #            else:
        for si, size in enumerate(self.sizevals):
            for oi, ori in enumerate(self.orivals):
                for ti, tf in enumerate(self.tfvals):
                    cell_trials[
                        oi, ti, si + 1, :, :
                    ] = self.mean_sweep_response[
                        (self.stim_table.Ori == ori)
                        & (self.stim_table.TF == tf)
                        & (self.stim_table.Size == size)
                        & (self.mean_sweep_eye.total < self.eye_thresh)
                    ].values[
                        :n_trials
                    ]

        for i in range(n_iter):
            for oi in range(len(self.orivals)):
                for ti in range(len(self.tfvals)):
                    for si in range(len(self.sizevals) + 1):
                        idx_1 = np.random.choice(
                            n_trials, int(n_trials / 2), replace=False
                        )
                        idx_2 = np.random.choice(
                            np.setdiff1d(range(n_trials), idx_1),
                            int(n_trials / 2),
                            replace=False,
                        )
                        response_first[oi, ti, si, :, i] = np.mean(
                            cell_trials[oi, ti, si, idx_1, :], axis=0
                        )
                        response_second[oi, ti, si, :, i] = np.mean(
                            cell_trials[oi, ti, si, idx_2, :], axis=0
                        )
        return response_first, response_second

    def get_osi(self, tuning):
        orivals_rad = np.deg2rad(self.orivals)
        tuning = np.where(tuning > 0, tuning, 0)
        CV_top_os = np.empty((8, tuning.shape[1]), dtype=np.complex128)
        for i in range(8):
            CV_top_os[i] = tuning[i] * np.exp(1j * 2 * orivals_rad[i])
        return np.abs(CV_top_os.sum(axis=0)) / tuning.sum(axis=0)

    def get_metrics(self):
        '''creates a table of metrics for each cell. We can make this more useful in the future

Returns
-------
metrics dataframe
        '''

        n_iter = 50
        n_trials = int(np.nanmin(self.response[:, :, 1:, :, 2]))
        print("Number of trials for cross-validation: " + str(n_trials))
        cell_index = np.array(range(self.numbercells))
        response_first, response_second = self.cross_validate_response(
            n_iter, n_trials
        )

        metrics = pd.DataFrame(
            columns=(
                'cell_index',
                'dir',
                'tf',
                'prefsize',
                'osi',
                'dsi',
                'dir_percent',
                'peak_mean',
                'peak_std',
                'blank_mean',
                'blank_std',
                'peak_percent_trials',
            ),
            index=cell_index,
        )
        metrics.cell_index = cell_index

        # cross-validated metrics
        DSI = pd.DataFrame(columns=cell_index.astype(str), index=range(n_iter))
        OSI = pd.DataFrame(columns=cell_index.astype(str), index=range(n_iter))
        DIR = pd.DataFrame(columns=cell_index.astype(str), index=range(n_iter))

        for ni in range(n_iter):
            # find pref direction for each cell for center only condition
            #            response_first = response_first[:,:,:,cell_index,:]
            #            response_second = response_second[:,:,:,cell_index,:]
            sort = np.where(
                response_first[:, :, :, :, ni]
                == np.nanmax(response_first[:, :, :, :, ni], axis=(0, 1, 2))
            )
            # TODO: this is where the TF is going to add issues...
            sortind = np.argsort(sort[3])
            pref_ori = sort[0][sortind]
            #            print(len(pref_ori))
            pref_tf = sort[1][sortind]
            pref_size = sort[2][sortind]
            cell_index = sort[3][sortind]
            inds = np.vstack((pref_ori, pref_tf, pref_size, cell_index))

            DIR.loc[ni] = pref_ori

            # osi
            OSI.loc[ni] = self.get_osi(
                response_second[:, inds[1], inds[2], inds[3], ni]
            )

            # dsi
            null_ori = np.mod(pref_ori + 4, 8)
            pref = response_second[inds[0], inds[1], inds[2], inds[3], ni]
            null = response_second[null_ori, inds[1], inds[2], inds[3], ni]
            null = np.where(null > 0, null, 0)
            DSI.loc[ni] = (pref - null) / (pref + null)

        metrics['osi'] = OSI.mean().values
        metrics['dsi'] = DSI.mean().values

        # how consistent is the selected preferred direction?
        for nc in range(self.numbercells):
            metrics['dir_percent'].loc[nc] = DIR[str(nc)].value_counts().max()

        # non cross-validated metrics
        cell_index = np.array(range(self.numbercells))
        sort = np.where(
            self.response[:, :, :, cell_index, 0]
            == np.nanmax(self.response[:, :, :, cell_index, 0], axis=(0, 1, 2))
        )
        sortind = np.argsort(sort[3])
        pref_ori = sort[0][sortind]
        pref_tf = sort[1][sortind]
        pref_size = sort[2][sortind]
        cell_index = sort[3][sortind]
        metrics['dir'] = pref_ori
        metrics['tf'] = pref_tf
        metrics['prefsize'] = pref_size
        metrics['peak_mean'] = self.response[
            pref_ori, pref_tf, pref_size, cell_index, 0
        ]
        metrics['peak_std'] = self.response[
            pref_ori, pref_tf, pref_size, cell_index, 1
        ]
        metrics['peak_percent_trials'] = self.response[
            pref_ori, pref_tf, pref_size, cell_index, 3
        ]
        metrics['blank_mean'] = self.response[0, 0, 0, cell_index, 0]
        metrics['blank_std'] = self.response[0, 0, 0, cell_index, 1]

        b = set(metrics.index)
        a = set(range(self.numbercells))
        toadd = a.difference(b)
        if len(toadd) > 0:
            newdf = pd.DataFrame(columns=metrics.columns, index=toadd)
            newdf.cell_index = toadd
            newdf.valid = False
            metrics = metrics.append(newdf)
            metrics.sort_index(inplace=True)

        metrics = metrics.join(self.roi[['cell_id', 'session_id', 'valid']])
        metrics['cre'] = self.cre
        metrics['area'] = self.area
        metrics['depth'] = self.depth

        return metrics, OSI, DSI, DIR

    def save_data(self):
        '''saves intermediate analysis files in an h5 file'''
        save_file = os.path.join(
            r'/Users/saskiad/Documents/Data/Openscope_Multiplex_trim/analysis/tf',
            str(self.session_id) + "_st_analysis.h5",
        )
        print("Saving data to: ", save_file)
        store = pd.HDFStore(save_file)
        store['sweep_response'] = self.sweep_response
        store['mean_sweep_response'] = self.mean_sweep_response
        store['sweep_p_values'] = self.sweep_p_values
        store['sweep_eye'] = self.sweep_eye
        store['mean_sweep_eye'] = self.mean_sweep_eye
        store['metrics'] = self.metrics
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response', data=self.response)
        f.close()


if __name__ == '__main__':
    #    expt_path = r'/Users/saskiad/Documents/Data/Openscope_Multiplex_trim/Size_Tuning_976843461_data.h5'
    #    eye_thresh = 10
    #    cre = 'test'
    #    area = 'area test'
    #    depth = '33'
    #    szt = SizeTuning(expt_path=expt_path, eye_thresh=eye_thresh, cre=cre, area=area, depth=depth)

    manifest = pd.read_csv(
        r'/Users/saskiad/Dropbox/Openscope Multiplex/data manifest.csv'
    )
    subset = manifest[manifest.Target == 'soma']
    print(len(subset))
    count = 0
    failed = []
    for index, row in subset.iterrows():
        if np.isfinite(row.Size_Tuning_Expt_ID):
            count += 1
            cre = row.Cre
            area = row.Area
            depth = row.Depth
            expt_path = (
                r'/Users/saskiad/Documents/Data/Openscope_Multiplex_trim/Size_Tuning_'
                + str(int(row.Size_Tuning_Expt_ID))
                + '_data.h5'
            )
            eye_thresh = 10
            try:
                szt = SizeTuning(
                    expt_path=expt_path,
                    eye_thresh=eye_thresh,
                    cre=cre,
                    area=area,
                    depth=depth,
                )
                if count == 1:
                    metrics_all = szt.metrics.copy()
                    print("reached here")
                else:
                    metrics_all = metrics_all.append(szt.metrics)
            except:
                print(expt_path + " FAILED")
                failed.append(int(row.Size_Tuning_Expt_ID))
