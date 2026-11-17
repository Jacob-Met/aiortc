import asyncio
import datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch

from aiortc.rtcdtlstransport import (
    SRTP_AEAD_AES_256_GCM,
    SRTP_AES128_CM_SHA1_80,
    RTCCertificate,
    RTCDtlsFingerprint,
    RTCDtlsParameters,
    RTCDtlsTransport,
    RtpRouter,
)
from aiortc.rtcrtpparameters import (
    RTCRtpCodecParameters,
    RTCRtpDecodingParameters,
    RTCRtpHeaderExtensionParameters,
    RTCRtpReceiveParameters,
    RTCRtpRtxParameters,
)
from aiortc.rtp import (
    RTCP_PSFB_APP,
    RTCP_PSFB_PLI,
    RTCP_RTPFB_NACK,
    AnyRtcpPacket,
    RtcpByePacket,
    RtcpPsfbPacket,
    RtcpReceiverInfo,
    RtcpRrPacket,
    RtcpRtpfbPacket,
    RtcpSenderInfo,
    RtcpSrPacket,
    RtpPacket,
    pack_remb_fci,
)
from OpenSSL import SSL

from .utils import asynctest, dummy_ice_transport_pair, load, set_loss_pattern

RTP = load("rtp.bin")
RTCP = load("rtcp_sr.bin")


class BrokenDataReceiver:
    async def _handle_data(self, data: bytes) -> None:
        raise Exception("some error")


class DummyDataReceiver:
    def __init__(self) -> None:
        self.data: list[bytes] = []

    async def _handle_data(self, data: bytes) -> None:
        self.data.append(data)


class DummyRtpReceiver:
    def __init__(self) -> None:
        self.rtp_packets: list[RtpPacket] = []
        self.rtcp_packets: list[AnyRtcpPacket] = []

    def _handle_disconnect(self) -> None:
        pass

    async def _handle_rtp_packet(self, packet: RtpPacket, arrival_time_ms: int) -> None:
        self.rtp_packets.append(packet)

    async def _handle_rtcp_packet(self, packet: AnyRtcpPacket) -> None:
        self.rtcp_packets.append(packet)


class DummyRtpSender:
    _ssrc = 0

    async def _handle_rtcp_packet(self, packet: AnyRtcpPacket) -> None:
        pass


class RTCCertificateTest(TestCase):
    def test_generate(self) -> None:
        certificate = RTCCertificate.generateCertificate()
        self.assertIsNotNone(certificate)

        expires = certificate.expires
        self.assertIsNotNone(expires)
        self.assertIsInstance(expires, datetime.datetime)

        fingerprints = certificate.getFingerprints()
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(fingerprints[0].algorithm, "sha-256")
        self.assertEqual(len(fingerprints[0].value), 95)
        self.assertEqual(fingerprints[1].algorithm, "sha-384")
        self.assertEqual(len(fingerprints[1].value), 143)
        self.assertEqual(fingerprints[2].algorithm, "sha-512")
        self.assertEqual(len(fingerprints[2].value), 191)


class RTCDtlsTransportTest(TestCase):
    def assertCounters(
        self,
        transport_a: RTCDtlsTransport,
        transport_b: RTCDtlsTransport,
        packets_sent_a: int,
        packets_sent_b: int,
    ) -> None:
        stats_a = transport_a._get_stats()[transport_a._stats_id]
        stats_b = transport_b._get_stats()[transport_b._stats_id]

        self.assertEqual(stats_a.packetsSent, packets_sent_a)
        self.assertEqual(stats_a.packetsReceived, packets_sent_b)
        self.assertGreater(stats_a.bytesSent, 0)
        self.assertGreater(stats_a.bytesReceived, 0)

        self.assertEqual(stats_b.packetsSent, packets_sent_b)
        self.assertEqual(stats_b.packetsReceived, packets_sent_a)
        self.assertGreater(stats_b.bytesSent, 0)
        self.assertGreater(stats_b.bytesReceived, 0)

        self.assertEqual(stats_a.bytesSent, stats_b.bytesReceived)
        self.assertEqual(stats_b.bytesSent, stats_a.bytesReceived)

    @asynctest
    async def test_data(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyDataReceiver()
        session1._register_data_receiver(receiver1)

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyDataReceiver()
        session2._register_data_receiver(receiver2)

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send encypted data
        await session1._send_data(b"ping")
        await asyncio.sleep(0.1)
        self.assertEqual(receiver2.data, [b"ping"])

        await session2._send_data(b"pong")
        await asyncio.sleep(0.1)
        self.assertEqual(receiver1.data, [b"pong"])

        # shutdown
        await session1.stop()
        await asyncio.sleep(0.1)
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

        # try closing again
        await session1.stop()
        await session2.stop()

        # try sending after close
        with self.assertRaises(ConnectionError):
            await session1._send_data(b"foo")

    @asynctest
    async def test_data_handler_error(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyDataReceiver()
        session1._register_data_receiver(receiver1)

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        session2._register_data_receiver(BrokenDataReceiver())

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send encypted data
        await session1._send_data(b"ping")
        await asyncio.sleep(0.1)

        # shutdown
        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_rtp(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyRtpReceiver()
        session1._register_rtp_receiver(
            receiver1,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=1831097322, payloadType=0)],
            ),
        )

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyRtpReceiver()
        session2._register_rtp_receiver(
            receiver2,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=4028317929, payloadType=0)],
            ),
        )

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertCounters(session1, session2, 2, 2)

        # send RTP
        await session1._send_rtp(RTP)
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 3, 2)
        self.assertEqual(len(receiver2.rtcp_packets), 0)
        self.assertEqual(len(receiver2.rtp_packets), 1)

        # send RTCP
        await session2._send_rtp(RTCP)
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 3, 3)
        self.assertEqual(len(receiver1.rtcp_packets), 1)
        self.assertEqual(len(receiver1.rtp_packets), 0)

        # shutdown
        await session1.stop()
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 4, 3)
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

        # try closing again
        await session1.stop()
        await session2.stop()

        # try sending after close
        with self.assertRaises(ConnectionError):
            await session1._send_rtp(RTP)

    @asynctest
    async def test_rtp_malformed(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        # receive truncated RTP
        await session1._handle_rtp_data(RTP[0:8], 0)

        # receive truncated RTCP
        await session1._handle_rtcp_data(RTCP[0:8])

    @asynctest
    async def test_srtp_unprotect_error(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyRtpReceiver()
        session1._register_rtp_receiver(
            receiver1,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=1831097322, payloadType=0)],
            ),
        )

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyRtpReceiver()
        session2._register_rtp_receiver(
            receiver2,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=4028317929, payloadType=0)],
            ),
        )

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send same RTP twice, to trigger error on the receiver side:
        # "replay check failed (bad index)"
        await session1._send_rtp(RTP)
        await session1._send_rtp(RTP)
        await asyncio.sleep(0.1)
        self.assertEqual(len(receiver2.rtcp_packets), 0)
        self.assertEqual(len(receiver2.rtp_packets), 1)

        # shutdown
        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_abrupt_disconnect(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # break connections -> tasks exits
        await transport1.stop()
        await transport2.stop()
        await asyncio.sleep(0.1)

        # close DTLS
        await session1.stop()
        await session2.stop()

        # check outcome
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

    @asynctest
    async def test_abrupt_disconnect_2(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        def fake_write_ssl() -> None:
            raise ConnectionError

        session1._write_ssl = fake_write_ssl  # type: ignore

        # close DTLS -> ConnectionError
        await session1.stop()
        await session2.stop()
        await asyncio.sleep(0.1)

        # check outcome
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

    @asynctest
    async def test_bad_client_fingerprint(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        bogus_parameters = RTCDtlsParameters(
            fingerprints=[
                RTCDtlsFingerprint(algorithm="sha-256", value="bogus_fingerprint")
            ]
        )
        await asyncio.gather(
            session1.start(bogus_parameters),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "connected")

        await session1.stop()
        await session2.stop()

    @patch("aiortc.rtcdtlstransport.SSL.Connection.do_handshake")
    @asynctest
    async def test_handshake_error(self, mock_do_handshake: MagicMock) -> None:
        mock_do_handshake.side_effect = SSL.Error(
            [("SSL routines", "", "decryption failed or bad record mac")]
        )

        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "failed")

        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_handshake_error_no_common_srtp_profile(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        session1._srtp_profiles = [SRTP_AEAD_AES_256_GCM]

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        session2._srtp_profiles = [SRTP_AES128_CM_SHA1_80]

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "failed")

        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_retransmit_after_handshake(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "connected")
        self.assertIsNone(session1._ssl.DTLSv1_get_timeout())

        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_lossy_channel(self) -> None:
        """
        Transport with 25% loss eventually connects.
        """
        transport1, transport2 = dummy_ice_transport_pair()
        loss_pattern = [True, False, False, False]
        set_loss_pattern(transport1, loss_pattern)
        set_loss_pattern(transport2, loss_pattern)

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        await session1.stop()
        await session2.stop()


class RtpRouterTest(TestCase):
    def test_route_rtcp(self) -> None:
        receiver = DummyRtpReceiver()
        sender = DummyRtpSender()

        router = RtpRouter()
        router.register_receiver(receiver, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_sender(sender, ssrc=3456)

        # BYE
        packet: AnyRtcpPacket = RtcpByePacket(sources=[1234, 2345])
        self.assertEqual(router.route_rtcp(packet), set([receiver]))

        # RR
        packet = RtcpRrPacket(
            ssrc=1234,
            reports=[
                RtcpReceiverInfo(
                    ssrc=3456,
                    fraction_lost=0,
                    packets_lost=0,
                    highest_sequence=630,
                    jitter=1906,
                    lsr=0,
                    dlsr=0,
                )
            ],
        )
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # SR
        packet = RtcpSrPacket(
            ssrc=1234,
            sender_info=RtcpSenderInfo(
                ntp_timestamp=0, rtp_timestamp=0, packet_count=0, octet_count=0
            ),
            reports=[
                RtcpReceiverInfo(
                    ssrc=3456,
                    fraction_lost=0,
                    packets_lost=0,
                    highest_sequence=630,
                    jitter=1906,
                    lsr=0,
                    dlsr=0,
                )
            ],
        )
        self.assertEqual(router.route_rtcp(packet), set([receiver, sender]))

        # PSFB - PLI
        packet = RtcpPsfbPacket(fmt=RTCP_PSFB_PLI, ssrc=1234, media_ssrc=3456)
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # PSFB - REMB
        packet = RtcpPsfbPacket(
            fmt=RTCP_PSFB_APP,
            ssrc=1234,
            media_ssrc=0,
            fci=pack_remb_fci(4160000, [3456]),
        )
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # PSFB - JUNK
        packet = RtcpPsfbPacket(fmt=RTCP_PSFB_APP, ssrc=1234, media_ssrc=0, fci=b"JUNK")
        self.assertEqual(router.route_rtcp(packet), set())

        # RTPFB
        packet = RtcpRtpfbPacket(fmt=RTCP_RTPFB_NACK, ssrc=1234, media_ssrc=3456)
        self.assertEqual(router.route_rtcp(packet), set([sender]))

    def test_route_rtp(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()

        router = RtpRouter()
        router.register_receiver(receiver1, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_receiver(receiver2, ssrcs=[3456, 4567], payload_types=[98, 99])

        # known SSRC and payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=1234, payload_type=96)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=2345, payload_type=97)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=3456, payload_type=98)), receiver2
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=4567, payload_type=99)), receiver2
        )

        # unknown SSRC, known payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=5678, payload_type=96)), receiver1
        )
        self.assertEqual(router.ssrc_table[5678], receiver1)

        # unknown SSRC and payload type
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=6789, payload_type=100)), None)

    def test_route_rtp_ambiguous_payload_type_mid(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()
        router = RtpRouter()
        router.register_receiver(receiver1, ssrcs=[1234], payload_types=[96], mid="0")
        router.register_receiver(receiver2, ssrcs=[2345], payload_types=[96], mid="1")

        # Without an SSRC association or MID, both tracks could decode PT 96.
        self.assertIsNone(router.route_rtp(RtpPacket(ssrc=3456, payload_type=96)))

        packet = RtpPacket(ssrc=3456, payload_type=96, sequence_number=100)
        packet.extensions.mid = "1"
        self.assertIs(router.route_rtp(packet), receiver2)
        self.assertIs(
            router.route_rtp(RtpPacket(ssrc=3456, payload_type=96)),
            receiver2,
        )

        # An unknown MID or a codec not negotiated on that MID cannot route.
        packet = RtpPacket(ssrc=4567, payload_type=96)
        packet.extensions.mid = "unnegotiated"
        self.assertIsNone(router.route_rtp(packet))
        packet.extensions.mid = "0"
        packet.payload_type = 98
        self.assertIsNone(router.route_rtp(packet))
        self.assertNotIn(4567, router.ssrc_table)

        # RFC 8843 section 9.2 permits a newer MID to update SSRC routing.
        packet = RtpPacket(ssrc=3456, payload_type=96, sequence_number=101)
        packet.extensions.mid = "0"
        self.assertIs(router.route_rtp(packet), receiver1)

        # A delayed packet with the older MID must not flip the association.
        packet.sequence_number = 99
        packet.extensions.mid = "1"
        self.assertIsNone(router.route_rtp(packet))
        self.assertIs(router.ssrc_table[3456], receiver1)

        # A MID update across a 16-bit sequence wrap must still win.
        packet = RtpPacket(ssrc=5678, payload_type=96, sequence_number=65534)
        packet.extensions.mid = "0"
        self.assertIs(router.route_rtp(packet), receiver1)
        packet.sequence_number = 65535
        self.assertIs(router.route_rtp(packet), receiver1)
        packet.sequence_number = 0
        packet.extensions.mid = "1"
        self.assertIs(router.route_rtp(packet), receiver2)
        packet.sequence_number = 65533
        packet.extensions.mid = "0"
        self.assertIsNone(router.route_rtp(packet))
        self.assertIs(router.ssrc_table[5678], receiver2)

    @asynctest
    async def test_route_rtp_wire_mid_bundle(self) -> None:
        transport = RTCDtlsTransport(
            dummy_ice_transport_pair()[0],
            [RTCCertificate.generateCertificate()],
        )
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()
        extension = RTCRtpHeaderExtensionParameters(
            id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"
        )
        codecs = [
            RTCRtpCodecParameters(
                mimeType="audio/opus", clockRate=48000, payloadType=96
            )
        ]
        for mid, ssrc, receiver in [
            ("0", 100, receiver1),
            ("1", 200, receiver2),
        ]:
            transport._register_rtp_receiver(
                receiver,
                RTCRtpReceiveParameters(
                    codecs=codecs,
                    headerExtensions=[extension],
                    muxId=mid,
                    encodings=[RTCRtpDecodingParameters(ssrc=ssrc, payloadType=96)],
                ),
            )

        # Parse actual wire-format RTP/MID extension and dispatch through
        # the DTLS transport, not just through the router directly.
        packet = RtpPacket(ssrc=300, payload_type=96, sequence_number=42)
        packet.extensions.mid = "1"
        await transport._handle_rtp_data(
            packet.serialize(transport._rtp_header_extensions_map),
            arrival_time_ms=1234,
        )
        self.assertEqual(receiver1.rtp_packets, [])
        self.assertEqual(len(receiver2.rtp_packets), 1)
        self.assertEqual(receiver2.rtp_packets[0].ssrc, 300)
        self.assertEqual(receiver2.rtp_packets[0].extensions.mid, "1")

    def test_register_rtp_receiver_with_rtx_ssrc(self) -> None:
        transport = RTCDtlsTransport(
            dummy_ice_transport_pair()[0],
            [RTCCertificate.generateCertificate()],
        )
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()
        codecs = [
            RTCRtpCodecParameters(
                mimeType="video/VP8", clockRate=90000, payloadType=96
            ),
            RTCRtpCodecParameters(
                mimeType="video/rtx",
                clockRate=90000,
                payloadType=97,
                parameters={"apt": 96},
            ),
        ]

        def parameters(mid: str, ssrc: int, rtx_ssrc: int) -> RTCRtpReceiveParameters:
            return RTCRtpReceiveParameters(
                codecs=codecs,
                muxId=mid,
                encodings=[
                    RTCRtpDecodingParameters(
                        ssrc=ssrc,
                        payloadType=96,
                        rtx=RTCRtpRtxParameters(ssrc=rtx_ssrc),
                    )
                ],
            )

        transport._register_rtp_receiver(receiver1, parameters("0", 100, 101))
        transport._register_rtp_receiver(receiver2, parameters("1", 200, 201))
        router = transport._rtp_router

        # Both video tracks use the same RTX PT, so RTX SSRC is required
        # even if retransmissions carry no MID extension.
        self.assertIs(router.route_rtp(RtpPacket(ssrc=101, payload_type=97)), receiver1)
        self.assertIs(router.route_rtp(RtpPacket(ssrc=201, payload_type=97)), receiver2)

        transport._register_rtp_receiver(receiver2, parameters("1", 300, 301))
        self.assertNotIn(201, router.ssrc_table)
        self.assertIs(router.route_rtp(RtpPacket(ssrc=301, payload_type=97)), receiver2)

    def test_register_receiver_replaces_mappings(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()
        router = RtpRouter()
        router.register_receiver(
            receiver1, ssrcs=[1234], payload_types=[96, 97], mid="0"
        )
        router.register_receiver(receiver2, ssrcs=[2345], payload_types=[96], mid="1")

        # A renegotiated SSRC and PT replace the old mappings without
        # touching another receiver sharing the same RTP payload type.
        router.register_receiver(receiver1, ssrcs=[3456], payload_types=[98], mid="0")
        self.assertNotIn(1234, router.ssrc_table)
        self.assertIs(router.ssrc_table[3456], receiver1)
        self.assertIs(router.ssrc_table[2345], receiver2)
        self.assertNotIn(97, router.payload_type_table)
        self.assertEqual(router.payload_type_table[96], {receiver2})
        self.assertEqual(router.payload_type_table[98], {receiver1})
        self.assertEqual(router.mid_table["0"], receiver1)
        self.assertEqual(router.mid_table["1"], receiver2)

        # RTCP sender reports follow the changed SSRC map as well.
        sender_info = RtcpSenderInfo(
            ntp_timestamp=0, rtp_timestamp=0, packet_count=0, octet_count=0
        )
        self.assertEqual(
            router.route_rtcp(RtcpSrPacket(ssrc=1234, sender_info=sender_info)), set()
        )
        self.assertEqual(
            router.route_rtcp(RtcpSrPacket(ssrc=3456, sender_info=sender_info)),
            {receiver1},
        )

    def test_route_rtp_ambiguous_payload_type(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()

        router = RtpRouter()
        router.register_receiver(receiver1, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_receiver(receiver2, ssrcs=[3456, 4567], payload_types=[96, 97])

        # known SSRC and payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=1234, payload_type=96)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=2345, payload_type=97)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=3456, payload_type=96)), receiver2
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=4567, payload_type=97)), receiver2
        )

        # unknown SSRC, ambiguous payload type
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=5678, payload_type=96)), None)
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=5678, payload_type=97)), None)
