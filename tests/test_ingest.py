"""Layer 0: ingest, format detection and native-text-layer classification.

The consequential test in this file is
``test_scanned_page_is_not_mistaken_for_native``.  Trusting a stray text layer
and skipping OCR is the single most damaging false positive in the library: it
produces a confident, empty result that looks like a successful extraction.
"""

from typing import Any

import pytest

import docpipe as dp
from conftest import requires_pymupdf

np = pytest.importorskip("numpy")


class TestSniffFormat:
    @pytest.mark.parametrize("data,expected", [
        (b"%PDF-1.7\n...", "pdf"),
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff\xe0", "jpeg"),
        (b"II*\x00", "tiff"),
        (b"MM\x00*", "tiff"),
        (b"BM\x00\x00", "bmp"),
        (b"GIF89a", "gif"),
        (b"From: a@b.com\r\nSubject: hi", "eml"),
    ])
    def test_magic_bytes(self, data, expected):
        assert dp.sniff_format(data) == expected

    def test_webp(self):
        assert dp.sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"

    def test_pdf_with_leading_junk(self):
        """Mail gateways and bad scanners prepend bytes to PDFs."""
        assert dp.sniff_format(b"\x00\x00garbage\n%PDF-1.4\n") == "pdf"

    def test_content_beats_the_extension(self, bill_image):
        """A JPEG named scan.pdf is a weekly occurrence on email attachments."""
        assert dp.sniff_format(dp.encode_jpeg(bill_image), "scan.pdf") == "jpeg"

    def test_extension_is_the_fallback_for_unknown_bytes(self):
        assert dp.sniff_format(b"\x01\x02\x03\x04", "x.tif") == "tiff"

    def test_unknown_stays_unknown(self):
        assert dp.sniff_format(b"\x01\x02\x03\x04", "x.bin") == "unknown"


class TestPageKindClassification:
    def test_plenty_of_text_is_native(self):
        assert dp.classify_page_kind(500, 0.0, False) is dp.PageKind.DIGITAL_NATIVE

    def test_text_over_a_large_image_is_hybrid(self):
        """A typed claim form with a photographed bill pasted in: both paths
        are needed, and treating it as native loses the insert."""
        assert dp.classify_page_kind(500, 0.8, True) is dp.PageKind.HYBRID

    def test_a_few_stray_characters_over_an_image_is_scanned(self):
        """A producer watermark must not buy a page a 'native' classification."""
        assert dp.classify_page_kind(9, 0.95, True) is dp.PageKind.SCANNED

    def test_nothing_at_all_is_blank(self):
        assert dp.classify_page_kind(0, 0.0, False) is dp.PageKind.BLANK

    def test_threshold_is_configurable(self):
        assert dp.classify_page_kind(20, 0.0, False, min_chars=10) \
            is dp.PageKind.DIGITAL_NATIVE


class TestIngestImage:
    def test_from_path(self, image_file):
        doc = dp.ingest(image_file)
        assert len(doc) == 1
        assert doc.pages[0].kind is dp.PageKind.SCANNED
        assert doc.source_uri == image_file

    def test_from_bytes(self, bill_image):
        doc = dp.ingest(dp.encode_png(bill_image), filename="scan.png")
        assert len(doc) == 1
        assert doc.source_uri == "scan.png"

    def test_from_file_object(self, image_file):
        with open(image_file, "rb") as fh:
            assert len(dp.ingest(fh)) == 1

    def test_geometry_follows_the_dpi(self, bill_image):
        at_150 = dp.ingest(dp.encode_png(bill_image), dpi=150)
        at_300 = dp.ingest(dp.encode_png(bill_image), dpi=300)
        assert at_150.pages[0].width_pt == pytest.approx(at_300.pages[0].width_pt * 2)

    def test_raster_is_available_immediately(self, image_file, bill_image):
        assert dp.ingest(image_file).pages[0].raster().shape == bill_image.shape

    def test_checksum_is_recorded(self, image_file):
        assert len(dp.ingest(image_file).meta["checksum"]) == 32

    def test_missing_file_raises(self):
        with pytest.raises(dp.IngestError):
            dp.ingest("/nonexistent/file.pdf")

    def test_empty_input_raises(self):
        with pytest.raises(dp.IngestError):
            dp.ingest(b"", filename="empty.png")

    def test_unrecognisable_bytes_raise(self):
        with pytest.raises(dp.IngestError):
            dp.ingest(b"\x00" * 64, filename="mystery.bin")


@requires_pymupdf
class TestIngestPDF:
    def test_native_pdf_keeps_its_text_layer(self, native_pdf):
        doc = dp.ingest(native_pdf)
        page = doc.pages[0]
        assert page.kind is dp.PageKind.DIGITAL_NATIVE
        assert "SUNRISE" in page.text()
        assert page.spans and page.spans[0].source == "pymupdf"

    def test_native_spans_carry_no_confidence(self, native_pdf):
        """A text layer is exact; a fabricated confidence would poison fusion."""
        assert all(s.confidence is None for s in dp.ingest(native_pdf).pages[0].spans)

    def test_scanned_page_is_not_mistaken_for_native(self, scanned_pdf):
        page = dp.ingest(scanned_pdf).pages[0]
        assert page.kind is dp.PageKind.SCANNED
        assert page.char_count == 0

    def test_rendering_is_lazy(self, native_pdf):
        page = dp.ingest(native_pdf).pages[0]
        assert page._raster is None
        assert page.has_raster()
        assert page.raster(150).shape[0] > 100

    def test_render_dpi_controls_pixel_size(self, scanned_pdf):
        low = dp.ingest(scanned_pdf).pages[0].raster(100)
        high = dp.ingest(scanned_pdf).pages[0].raster(200)
        assert high.shape[0] == pytest.approx(low.shape[0] * 2, abs=3)

    def test_page_geometry_is_in_points(self, native_pdf):
        page = dp.ingest(native_pdf).pages[0]
        assert page.width_pt == pytest.approx(595, abs=1)
        assert page.height_pt == pytest.approx(842, abs=1)

    def test_max_pages_truncates_and_warns(self, tmp_path):
        fitz: Any = dp._fitz()
        pdf = fitz.open()
        for _ in range(5):
            pdf.new_page()
        path = str(tmp_path / "many.pdf")
        pdf.save(path)
        doc = dp.ingest(path, max_pages=2)
        assert len(doc) == 2
        assert any("truncated" in w for w in doc.warnings)

    def test_page_range_selects_specific_pages(self, tmp_path):
        fitz: Any = dp._fitz()
        pdf = fitz.open()
        for i in range(4):
            page = pdf.new_page()
            page.insert_text((50, 50), "page number %d" % i, fontsize=14)
        path = str(tmp_path / "ranged.pdf")
        pdf.save(path)
        doc = dp.ingest_pdf(path, page_range=[1, 3])
        assert [p.index for p in doc.pages] == [1, 3]

    def test_metadata_is_captured(self, native_pdf):
        meta = dp.ingest(native_pdf).meta
        assert meta["format"] == "pdf" and meta["page_count"] == 1

    def test_corrupt_pdf_raises_ingest_error(self):
        with pytest.raises(dp.IngestError):
            dp.ingest_pdf(b"%PDF-1.4\nthis is not a pdf at all")

    def test_encrypted_pdf_opens_with_an_empty_password(self, tmp_path):
        fitz: Any = dp._fitz()
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 50), "secret hospital bill total 500", fontsize=12)
        path = str(tmp_path / "locked.pdf")
        pdf.save(path, encryption=fitz.PDF_ENCRYPT_AES_256,
                 owner_pw="owner", permissions=int(fitz.PDF_PERM_ACCESSIBILITY))
        doc = dp.ingest(path)
        assert "secret" in doc.text()


@pytest.mark.skipif(not dp.have("PIL"), reason="needs Pillow")
class TestIngestTIFF:
    def test_multi_page_tiff(self, tmp_path, bill_image):
        from PIL import Image
        frames = [Image.fromarray(bill_image), Image.fromarray(bill_image)]
        path = str(tmp_path / "fax.tiff")
        frames[0].save(path, save_all=True, append_images=frames[1:])
        doc = dp.ingest(path)
        assert len(doc) == 2
        assert [p.index for p in doc.pages] == [0, 1]


class TestIngestEmail:
    def _message(self, attachments, body="Please find the bill attached."):
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = "Claim 47812"
        msg["From"] = "clerk@hospital.example"
        msg["To"] = "claims@insurer.example"
        msg.set_content(body)
        for name, data, (maintype, subtype) in attachments:
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
        return msg.as_bytes()

    def test_attachments_become_pages(self, bill_image):
        raw = self._message([("bill.png", dp.encode_png(bill_image), ("image", "png"))])
        doc = dp.ingest(raw, filename="claim.eml")
        pages = [p for p in doc.pages if p.meta.get("attachment") == "bill.png"]
        assert len(pages) == 1
        assert doc.meta["subject"] == "Claim 47812"

    def test_body_becomes_a_synthetic_text_page(self, bill_image):
        raw = self._message([("bill.png", dp.encode_png(bill_image), ("image", "png"))],
                            body="Total payable is 118250.00 rupees.")
        doc = dp.ingest(raw, filename="claim.eml")
        body_pages = [p for p in doc.pages if p.meta.get("synthetic")]
        assert len(body_pages) == 1
        assert "118250.00" in body_pages[0].text()

    def test_pages_are_renumbered_across_attachments(self, bill_image):
        raw = self._message([
            ("a.png", dp.encode_png(bill_image), ("image", "png")),
            ("b.png", dp.encode_png(bill_image), ("image", "png")),
        ])
        doc = dp.ingest(raw, filename="claim.eml")
        assert [p.index for p in doc.pages] == list(range(len(doc.pages)))

    def test_undecodable_attachment_is_warned_about_not_fatal(self, bill_image):
        raw = self._message([
            ("broken.png", b"\x00\x01\x02not-an-image", ("application", "octet-stream")),
            ("good.png", dp.encode_png(bill_image), ("image", "png")),
        ])
        doc = dp.ingest(raw, filename="claim.eml")
        assert any("skipped attachment" in w for w in doc.warnings)
        assert any(p.meta.get("attachment") == "good.png" for p in doc.pages)


class TestDirectoryAndMerge:
    def test_ingest_dir_reads_everything(self, tmp_path, bill_image):
        for i in range(3):
            dp.save_image(bill_image, str(tmp_path / ("scan%d.png" % i)))
        assert len(dp.ingest_dir(str(tmp_path))) == 3

    def test_ingest_dir_reports_failures_without_aborting(self, tmp_path, bill_image):
        dp.save_image(bill_image, str(tmp_path / "good.png"))
        with open(str(tmp_path / "bad.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nnot really")
        docs = dp.ingest_dir(str(tmp_path))
        assert len(docs) == 2
        assert sum(1 for d in docs if d.warnings) == 1

    def test_ingest_dir_pattern_filters(self, tmp_path, bill_image):
        dp.save_image(bill_image, str(tmp_path / "keep.png"))
        dp.save_image(bill_image, str(tmp_path / "skip.jpg"))
        assert len(dp.ingest_dir(str(tmp_path), pattern=r"\.png$")) == 1

    def test_merge_renumbers_pages_and_their_bboxes(self, bill_image):
        a = dp.document_from_images([bill_image], dpi=150)
        b = dp.document_from_images([bill_image], dpi=150)
        b.pages[0].spans = [dp.TextSpan("x", dp.BBox(0, 1, 2, 3, 4))]
        merged = dp.merge_documents([a, b])
        assert [p.index for p in merged.pages] == [0, 1]
        assert merged.pages[1].spans[0].bbox.page == 1

    def test_document_from_images_helper(self, bill_image):
        doc = dp.document_from_images([bill_image, bill_image], dpi=200)
        assert len(doc) == 2 and doc.pages[1].raster_dpi == 200
