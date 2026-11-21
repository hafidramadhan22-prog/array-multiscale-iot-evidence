"""Prior-only host context for the Weighted Voting development experiment.

The representation is a controlled extension, not a claim of algorithmic novelty.
Endpoint identifiers select state only and never appear in feature values.
"""
from __future__ import annotations

import numpy as np

from sbfe import CausalRing, WINDOW_NAMES

WINDOWS = (.1, 1.)
HOST_NAMES = (
    "log_packets", "log_bytes", "log_peers", "log_local_tcp_ports",
    "log_peer_tcp_ports", "mean_bytes", "std_bytes", "syn_fraction",
    "ack_fraction", "rst_fraction",
)
PRIOR_DESTINATION = [f"prior_destination_{int(w * 1000)}ms_{name}"
                     for w in WINDOWS for name in WINDOW_NAMES]
PRIOR_SOURCE = [f"prior_source_{int(w * 1000)}ms_{name}"
                for w in WINDOWS for name in HOST_NAMES]
PRIOR_SYMMETRIC = [f"prior_endpoint_{rank}_{int(w * 1000)}ms_{name}"
                   for rank in ("higher", "lower") for w in WINDOWS for name in HOST_NAMES]


class PriorEndpointRing(CausalRing):
    """Bounded shared packet history, queried before inserting the current packet.

    query_prior is read-only. Windows use max(previous arrival clock, current
    timestamp), so backwards timestamps do not reorder the observed stream.
    Source/destination query exchange leaves the symmetric representation equal.
    """

    def _records_before(self, timestamp):
        clock = max(int(timestamp), self.clock if self.clock is not None else int(timestamp))
        if not self.count:
            return self.data[:0], clock
        if self.start + self.count <= self.capacity:
            records = self.data[self.start:self.start + self.count]
        else:
            records = np.concatenate([
                self.data[self.start:], self.data[:(self.start + self.count) % self.capacity]
            ])
        return records[records[:, 0] >= clock - int(max(self.windows) * 1e9)], clock

    def _host_vector(self, records, clock, host):
        # Zero means an unavailable network endpoint in the inherited decoder.
        # Do not merge all non-IP packets into a fictitious common host.
        if not host:
            return np.zeros(len(self.windows) * len(HOST_NAMES), dtype=float)
        involved = records[(records[:, 1] == host) | (records[:, 2] == host)]
        out = []
        for window in self.windows:
            r = involved[involved[:, 0] >= clock - int(window * 1e9)]
            if not len(r):
                out.extend([0.] * len(HOST_NAMES))
                continue
            source = r[:, 1] == host
            peers = np.where(source, r[:, 2], r[:, 1])
            tcp = r[:, 7] == 6
            local_ports = np.where(source, r[:, 3], r[:, 4])[tcp]
            peer_ports = np.where(source, r[:, 4], r[:, 3])[tcp]
            flags = r[:, 6]
            out.extend([
                np.log1p(len(r)), np.log1p(r[:, 5].sum()), np.log1p(len(np.unique(peers))),
                np.log1p(len(np.unique(local_ports))), np.log1p(len(np.unique(peer_ports))),
                r[:, 5].mean(), r[:, 5].std(), np.mean((flags & 2) != 0),
                np.mean((flags & 16) != 0), np.mean((flags & 4) != 0),
            ])
        return np.asarray(out, dtype=float)

    def _destination_vector(self, records, clock, dst):
        if not dst:
            return np.zeros(len(self.windows) * len(WINDOW_NAMES), dtype=float)
        out = []
        involved = records[records[:, 2] == dst]
        for window in self.windows:
            r = involved[involved[:, 0] >= clock - int(window * 1e9)]
            if not len(r):
                out.extend([0.] * len(WINDOW_NAMES))
                continue
            flags = r[:, 6]
            out.extend([
                np.log1p(len(r)), np.log1p(r[:, 5].sum()),
                np.log1p(len(np.unique(r[:, 1]))), np.log1p(len(np.unique(r[:, 4]))),
                r[:, 5].mean(), r[:, 5].std(), np.mean((flags & 2) != 0),
                np.mean((flags & 16) != 0), np.mean((flags & 4) != 0),
                len(records) / self.capacity,
            ])
        return np.asarray(out, dtype=float)

    @staticmethod
    def _activity_key(vector):
        # Prefer the longest-window packet/byte counts. Full numeric tie break
        # makes order deterministic and never falls back to an endpoint ID.
        return tuple(vector[-len(HOST_NAMES):]) + tuple(vector)

    def query_prior(self, timestamp, src, dst):
        records, clock = self._records_before(timestamp)
        source = self._host_vector(records, clock, src)
        destination = self._host_vector(records, clock, dst)
        endpoints = sorted((source, destination), key=self._activity_key, reverse=True)
        return {
            "prior_destination": self._destination_vector(records, clock, dst),
            "prior_source": source,
            "prior_symmetric": np.concatenate(endpoints),
        }


def feature_schema():
    return {
        "prior_destination": PRIOR_DESTINATION,
        "prior_source": PRIOR_SOURCE,
        "prior_symmetric": PRIOR_SYMMETRIC,
    }
