"""Image primitives, and OpenCV / NumPy-fallback parity.

The fallbacks exist so the library works in deployments where an OpenCV wheel
is not available.  That promise is only worth anything if the two paths agree,
so most of these tests run under ``both_image_backends`` and several assert
numerical closeness between them directly.
"""

import pytest

import docpipe as dp

np = pytest.importorskip("numpy")


@pytest.fixture
def gradient():
    x = np.linspace(0, 255, 120)
    return np.tile(x, (80, 1)).astype(np.uint8)


@pytest.fixture
def bars():
    img = np.full((200, 300), 255, np.uint8)
    img[40:60, 20:280] = 0
    img[100:120, 20:200] = 0
    img[150:170, 20:250] = 30
    return img


class TestConversions:
    def test_to_gray_from_rgb(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[:, :, 0] = 255            # pure red
        gray = dp.to_gray(rgb)
        assert gray.ndim == 2
        assert gray[0, 0] == pytest.approx(76, abs=1)   # 0.299 * 255

    def test_to_gray_is_identity_for_gray(self, bars):
        assert dp.to_gray(bars) is bars

    def test_to_rgb_broadcasts_gray(self, bars):
        rgb = dp.to_rgb(bars)
        assert rgb.shape == bars.shape + (3,)
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_alpha_channel_is_dropped(self):
        rgba = np.zeros((4, 4, 4), np.uint8)
        assert dp.to_gray(rgba).shape == (4, 4)

    def test_ensure_uint8_scales_unit_floats(self):
        out = dp.ensure_uint8(np.full((2, 2), 0.5, np.float32))
        assert out.dtype == np.uint8 and out[0, 0] == 127

    def test_ensure_uint8_clips_out_of_range(self):
        out = dp.ensure_uint8(np.array([[-40, 300]], np.int16))
        assert out[0, 0] == 0 and out[0, 1] == 255

    def test_ensure_uint8_maps_bools_to_full_range(self):
        out = dp.ensure_uint8(np.array([[True, False]]))
        assert out[0, 0] == 255 and out[0, 1] == 0


class TestResize:
    def test_scale_and_explicit_size(self, both_image_backends, bars):
        assert dp.resize_image(bars, scale=0.5).shape == (100, 150)
        assert dp.resize_image(bars, size=(60, 40)).shape == (40, 60)

    def test_identity_is_a_no_op(self, bars):
        assert dp.resize_image(bars, scale=1.0) is bars

    def test_colour_images_keep_their_channels(self, both_image_backends):
        rgb = np.zeros((40, 60, 3), np.uint8)
        assert dp.resize_image(rgb, scale=2.0).shape == (80, 120, 3)

    def test_backends_agree(self, bars):
        dp.set_opencv_enabled(True)
        with_cv = dp.resize_image(bars, scale=0.5)
        with dp.without_opencv():
            without_cv = dp.resize_image(bars, scale=0.5)
        assert with_cv.shape == without_cv.shape
        # Different interpolation kernels; the structure must still match.
        assert float(np.mean(np.abs(with_cv.astype(float) - without_cv.astype(float)))) < 12.0


class TestRotate:
    def test_no_op_below_threshold(self, bars):
        assert dp.rotate_image(bars, 0.0) is bars

    def test_expand_grows_the_canvas(self, both_image_backends, bars):
        out = dp.rotate_image(bars, 30.0, border_value=255, expand=True)
        assert out.shape[0] > bars.shape[0] and out.shape[1] > bars.shape[1]

    def test_no_expand_keeps_the_canvas(self, both_image_backends, bars):
        assert dp.rotate_image(bars, 10.0, expand=False).shape == bars.shape

    def test_border_defaults_to_the_median_not_black(self, both_image_backends):
        """Black wedges would corrupt every downstream ink measurement."""
        white = np.full((60, 60), 255, np.uint8)
        out = dp.rotate_image(white, 20.0, expand=True)
        assert out[0, 0] == 255

    def test_positive_angle_rotates_counter_clockwise_in_both_backends(self):
        """OpenCV's convention is the contract; the fallback must match it."""
        img = np.full((80, 80), 255, np.uint8)
        img[10:20, 30:50] = 0          # a bar near the top

        dp.set_opencv_enabled(True)
        rotated_cv = dp.rotate_image(img, 90.0, border_value=255, expand=False)
        with dp.without_opencv():
            rotated_np = dp.rotate_image(img, 90.0, border_value=255, expand=False)

        def centroid(a):
            ys, xs = np.nonzero(a < 128)
            return (float(ys.mean()), float(xs.mean()))

        cy_cv, cx_cv = centroid(rotated_cv)
        cy_np, cx_np = centroid(rotated_np)
        assert abs(cy_cv - cy_np) < 3.0 and abs(cx_cv - cx_np) < 3.0
        # Top bar rotated CCW ends up on the left.
        assert cx_cv < 40.0


class TestBlurAndThreshold:
    def test_gaussian_blur_reduces_gradient_energy(self, both_image_backends, bars):
        sharp = float(dp.gradient_magnitude(bars).mean())
        soft = float(dp.gradient_magnitude(dp.gaussian_blur(bars, 3.0)).mean())
        assert soft < sharp

    @pytest.mark.skipif(not dp.have("cv2"), reason="needs OpenCV for comparison")
    @pytest.mark.parametrize("sigma", [0.8, 1.2, 2.0, 3.0, 4.0])
    def test_fallback_blurs_by_the_same_amount_as_opencv(self, bill_image, sigma):
        """Three box filters only approximate a Gaussian if their widths are
        variance-matched.  Choosing the radius by eye over-blurs by roughly a
        factor of two, which silently makes every fallback pipeline softer than
        the OpenCV one it stands in for."""
        dp.set_opencv_enabled(True)
        with_cv = dp.measure_blur(dp.gaussian_blur(bill_image, sigma))
        with dp.without_opencv():
            without_cv = dp.measure_blur(dp.gaussian_blur(bill_image, sigma))
        assert without_cv == pytest.approx(with_cv, abs=0.05)

    def test_box_widths_approach_the_requested_variance(self):
        """Box widths must be odd integers, so three of them can only land on
        a discrete set of variances -- the match is coarse at small sigma and
        tightens as sigma grows.  The binding guarantee is the measured-parity
        test above; this one pins the construction itself."""
        ratios = []
        for sigma in (1.0, 2.0, 3.0, 5.0, 8.0):
            widths = dp._box_widths_for_gaussian(sigma)
            assert len(widths) == 3
            assert all(w % 2 == 1 and w >= 1 for w in widths)
            variance = sum((w * w - 1) / 12.0 for w in widths)
            ratios.append(variance / (sigma * sigma))
            assert variance == pytest.approx(sigma * sigma, rel=0.4)
        # Quantisation error must shrink, not grow, with sigma.
        assert ratios == sorted(ratios)

    def test_zero_sigma_is_a_no_op(self, bars):
        assert np.array_equal(dp.gaussian_blur(bars, 0), bars)

    def test_median_blur_removes_salt_and_pepper(self, both_image_backends):
        img = np.full((60, 60), 255, np.uint8)
        rng = np.random.RandomState(0)
        ys, xs = rng.randint(0, 60, 40), rng.randint(0, 60, 40)
        img[ys, xs] = 0
        cleaned = dp.median_blur(img, 3)
        assert (cleaned < 128).sum() < (img < 128).sum() / 4

    def test_box_blur_of_a_constant_image_is_constant(self, both_image_backends):
        img = np.full((40, 40), 120, np.uint8)
        assert abs(int(dp.box_blur(img, 3).mean()) - 120) <= 1

    def test_otsu_returns_the_centre_of_the_optimal_valley(self, both_image_backends):
        """Every threshold in the empty valley is equally optimal; the centre
        is the one that survives sensor noise around either mode."""
        img = np.concatenate([np.full((50, 50), 30, np.uint8),
                              np.full((50, 50), 220, np.uint8)])
        assert 60 < dp.otsu_threshold(img) < 200

    def test_otsu_of_a_flat_image_is_a_safe_midpoint(self, both_image_backends):
        assert dp.otsu_threshold(np.full((20, 20), 77, np.uint8)) == 128

    def test_ink_mask_marks_dark_pixels(self, bars):
        mask = dp.ink_mask(bars)
        assert mask[50, 100] and not mask[5, 5]

    def test_otsu_backends_agree(self, bars):
        dp.set_opencv_enabled(True)
        with_cv = dp.otsu_threshold(bars)
        with dp.without_opencv():
            without_cv = dp.otsu_threshold(bars)
        assert with_cv == without_cv     # pure NumPy in both paths


class TestWindowStats:
    def test_mean_of_a_constant_image(self, gradient):
        flat = np.full((30, 30), 100, np.float64)
        mean, std = dp.window_stats(flat, 5)
        assert np.allclose(mean, 100.0)
        assert np.allclose(std, 0.0)

    def test_border_windows_are_not_biased_toward_zero(self):
        """Zero-padding would make every border pixel look dark."""
        flat = np.full((20, 20), 200, np.float64)
        mean, _ = dp.window_stats(flat, 9)
        assert mean[0, 0] == pytest.approx(200.0)

    def test_std_tracks_local_variation(self):
        img = np.zeros((40, 40), np.float64)
        img[:, 20:] = 255
        _, std = dp.window_stats(img, 5)
        assert std[20, 20] > 50.0     # at the edge
        assert std[20, 5] == pytest.approx(0.0)


class TestMorphology:
    def test_erode_shrinks_and_dilate_grows(self, both_image_backends):
        mask = np.zeros((40, 40), bool)
        mask[15:25, 15:25] = True
        assert dp.morph_binary(mask, 3, 3, "erode").sum() < mask.sum()
        assert dp.morph_binary(mask, 3, 3, "dilate").sum() > mask.sum()

    def test_open_removes_isolated_specks(self, both_image_backends):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        mask[2, 2] = True
        opened = dp.morph_binary(mask, 3, 3, "open")
        assert not opened[2, 2]
        assert opened[20, 20]

    def test_long_kernel_isolates_horizontal_runs(self, both_image_backends):
        mask = np.zeros((60, 200), bool)
        mask[30, 10:190] = True      # a rule
        mask[10:20, 100] = True      # a glyph stroke
        rules = dp.morph_binary(mask, 1, 80, "open")
        assert rules[30, 100]
        assert not rules[15, 100]

    def test_unknown_operation_is_rejected(self):
        with pytest.raises(dp.ConfigError):
            dp.morph_binary(np.zeros((4, 4), bool), 3, 3, "nonsense")


class TestCodecs:
    def test_png_round_trip_is_lossless(self, bars):
        assert np.array_equal(dp.decode_image(dp.encode_png(bars)), bars)

    def test_jpeg_round_trip_is_close(self, bars):
        decoded = dp.decode_image(dp.encode_jpeg(bars, quality=95))
        assert decoded.shape[:2] == bars.shape[:2]
        assert float(np.mean(np.abs(dp.to_gray(decoded).astype(float)
                                    - bars.astype(float)))) < 12.0

    def test_sniff_recognises_what_we_write(self, bars):
        assert dp.sniff_format(dp.encode_png(bars)) == "png"
        assert dp.sniff_format(dp.encode_jpeg(bars)) == "jpeg"

    def test_save_image_picks_the_codec_from_the_extension(self, tmp_path, bars):
        jpg = dp.save_image(bars, str(tmp_path / "x.jpg"))
        png = dp.save_image(bars, str(tmp_path / "x.png"))
        with open(jpg, "rb") as fh:
            assert dp.sniff_format(fh.read()) == "jpeg"
        with open(png, "rb") as fh:
            assert dp.sniff_format(fh.read()) == "png"


class TestProjection:
    def test_zero_angle_is_the_plain_row_sum(self, bars):
        assert np.allclose(dp.row_projection_at_angle(bars, 0.0), bars.sum(axis=1))

    def test_shearing_columns_actually_changes_the_profile(self, bars):
        """A horizontal row shear cannot change row sums -- this must."""
        straight = dp.row_projection_at_angle(dp.ink_mask(bars).astype(float), 0.0)
        sheared = dp.row_projection_at_angle(dp.ink_mask(bars).astype(float), 8.0)
        assert not np.allclose(straight, sheared)
        assert float(np.var(sheared)) < float(np.var(straight))


class TestBackgroundEstimation:
    def test_background_erases_text_but_keeps_illumination(self, both_image_backends):
        img = np.full((200, 200), 240, np.uint8)
        img[:, 100:] = 160                     # a lighting step
        img[50:60, 20:180] = 0                 # a line of text
        background = dp.estimate_background(img)
        assert background[55, 100] > 100       # text gone
        assert background[10, 20] > background[10, 180]   # step preserved


def test_capabilities_reports_booleans():
    caps = dp.capabilities()
    assert set(map(type, caps.values())) == {bool}
    assert "numpy" in caps and "opencv_enabled" in caps


def test_without_opencv_restores_the_previous_setting():
    dp.set_opencv_enabled(True)
    with dp.without_opencv():
        assert dp._cv2() is None
    assert (dp._cv2() is not None) == dp.have("cv2")
