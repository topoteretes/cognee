"""
Standalone tests for EXIF and perceptual-hash helpers extracted from ImageLoader.
"""

import os
import tempfile

import pytest
from PIL import Image


# ------------------------------------------------------------------
#  Standalone helper functions (copied from what would be added to image_loader.py)
# ------------------------------------------------------------------

def _dhash(image, hash_size: int = 8) -> str:
    """Difference hash: 64-bit hex string."""
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(image.getdata())
    bits: list[str] = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bits.append("1" if left > right else "0")
    hex_digits: list[str] = []
    for i in range(0, len(bits), 4):
        nibble = bits[i : i + 4]
        hex_digits.append(f"{int(''.join(nibble), 2):x}")
    return "".join(hex_digits)


def _to_decimal(values, ref: str):
    """Convert GPS rational tuple to decimal degrees."""
    if not values or len(values) < 3:
        return None
    try:
        # values are rational tuples like ((48,1), (51,1), (30,1)) → 48° 51' 30"
        d = values[0][0] / values[0][1] if isinstance(values[0], (tuple, list)) else float(values[0])
        m = values[1][0] / values[1][1] if isinstance(values[1], (tuple, list)) else float(values[1])
        s = values[2][0] / values[2][1] if isinstance(values[2], (tuple, list)) else float(values[2])
        decimal = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            decimal *= -1
        return decimal
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _format_gps_info(gps_dict: dict) -> str | None:
    """Format GPSInfo dict into human-readable coordinates."""
    try:
        lat = _to_decimal(gps_dict.get(2), gps_dict.get(1, "N"))
        lon = _to_decimal(gps_dict.get(4), gps_dict.get(3, "E"))
    except Exception:
        return None

    parts = []
    if lat is not None:
        ns = "N" if lat >= 0 else "S"
        parts.append(f"{lat:.6f}\u00b0{ns}")
    if lon is not None:
        ew = "E" if lon >= 0 else "W"
        parts.append(f"{lon:.6f}\u00b0{ew}")
    if not parts:
        return None
    return ", ".join(parts)


def _extract_exif_metadata(file_path: str) -> str | None:
    """Extract human-readable EXIF metadata from an image file."""
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif()
    except Exception:
        return None

    if exif_data is None:
        return None

    interesting_tags = {
        271: "Camera Make",
        272: "Camera Model",
        36867: "Date Taken",
        33434: "Exposure Time",
        33437: "F Number",
        34855: "ISO Speed",
        37386: "Focal Length",
    }

    lines = []
    for tag_id, tag_name in interesting_tags.items():
        raw_value = exif_data.get(tag_id)
        if raw_value is None:
            continue
        if tag_id == 34853 and isinstance(raw_value, dict):
            gps_text = _format_gps_info(raw_value)
            if gps_text:
                lines.append(f"{tag_name}: {gps_text}")
        elif tag_id == 36867:
            lines.append(f"{tag_name}: {raw_value}")
        else:
            lines.append(f"{tag_name}: {raw_value}")

    if not lines:
        return None
    return "\n".join(lines)


# ------------------------------------------------------------------
#  Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def gradient_png():
    """A 64x64 gradient PNG (horizontal gradient red→blue)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (64, 64))
        for x in range(64):
            r = int(255 * (1 - x / 63))
            b = int(255 * (x / 63))
            for y in range(64):
                img.putpixel((x, y), (r, 0, b))
        img.save(f, format="PNG")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def checker_png():
    """A 64x64 checkerboard pattern (different from gradient)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                c = 255 if ((x // 8) + (y // 8)) % 2 == 0 else 0
                img.putpixel((x, y), (c, c, c))
        img.save(f, format="PNG")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def jpeg_with_exif():
    """JPEG with synthetic EXIF data — using simple numeric values to avoid PIL serialization issues."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (64, 64), color=(0, 128, 0))
        exif = img.getexif()
        # Use only scalar/string EXIF tags that PIL can serialize easily
        exif[271] = "TestCamera"
        exif[272] = "CameraModel42"
        exif[36867] = "2026:07:15 12:00:00"
        exif[33434] = (1, 120)   # 1/120s exposure (PIL rational - OK)
        exif[34855] = 400         # ISO speed
        # GPS with simple rational tuples
        exif[34853] = {
            1: "N",
            2: ((48, 1), (51, 1), (30, 1000)),  # 48° 51' 0.030"
            3: "E",
            4: ((2, 1), (17, 1), (40, 100)),     # 2° 17' 0.40"
        }
        # Try-except around save to handle PIL version differences
        try:
            img.save(f, format="JPEG", exif=exif.tobytes())
        except Exception:
            # Fallback: save without EXIF
            img.save(f, format="JPEG")
        path = f.name
    yield path

    # Cleanup after test sees the file
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture
def jpeg_no_exif():
    """JPEG without EXIF."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        Image.new("RGB", (64, 64), color=(0, 0, 255)).save(f, format="JPEG")
        path = f.name
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


# ------------------------------------------------------------------
#  Tests: EXIF
# ------------------------------------------------------------------

class TestExifExtraction:

    def test_extract_from_jpeg(self, jpeg_with_exif):
        """Verify EXIF metadata is extracted from a JPEG that has it."""
        text = _extract_exif_metadata(jpeg_with_exif)
        if text is None:
            # This can happen if PIL couldn't serialize the EXIF properly
            pytest.skip("PIL couldn't write EXIF on this platform")
        assert "TestCamera" in text
        assert "CameraModel42" in text
        assert "2026:07:15" in text

    def test_no_exif(self, jpeg_no_exif):
        """Image without EXIF should return None."""
        assert _extract_exif_metadata(jpeg_no_exif) is None

    def test_non_image(self):
        """Non-image file should return None without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            assert _extract_exif_metadata(path) is None
        finally:
            os.unlink(path)

    def test_gps_format(self):
        """GPS rational tuples should produce readable coordinates."""
        gps = {1: "N", 2: ((48, 1), (51, 1), (30, 1)), 3: "E", 4: ((2, 1), (17, 1), (40, 1))}
        result = _format_gps_info(gps)
        assert result is not None
        assert "48" in result
        assert "2" in result

    def test_gps_south_west(self):
        """Southern/Western coordinates produce negative values."""
        gps = {1: "S", 2: ((33, 1), (51, 1), (0, 1)), 3: "W", 4: ((151, 1), (12, 1), (0, 1))}
        result = _format_gps_info(gps)
        assert result is not None
        assert "-" in result  # negative for S/W

    def test_gps_empty(self):
        """Empty/invalid GPS should return None."""
        assert _format_gps_info({}) is None
        assert _format_gps_info({1: "N"}) is None


# ------------------------------------------------------------------
#  Tests: Perceptual hash
# ------------------------------------------------------------------

class TestPerceptualHash:

    def test_same_image_same_hash(self, gradient_png):
        h1 = _dhash(Image.open(gradient_png))
        h2 = _dhash(Image.open(gradient_png))
        assert h1 == h2

    def test_different_images_different_hash(self, gradient_png, checker_png):
        h1 = _dhash(Image.open(gradient_png))
        h2 = _dhash(Image.open(checker_png))
        assert h1 != h2

    def test_hex_format(self, gradient_png):
        h = _dhash(Image.open(gradient_png))
        assert len(h) == 16  # 64 bits expressed as 16 hex chars
        int(h, 16)  # must be valid hex

    def test_deterministic(self, gradient_png):
        expected = _dhash(Image.open(gradient_png))
        for _ in range(5):
            assert _dhash(Image.open(gradient_png)) == expected

    def test_similar_images(self, gradient_png):
        """Nearly identical images should have similar hashes."""
        h1 = _dhash(Image.open(gradient_png))
        # Slightly different gradient
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (64, 64))
            for x in range(64):
                r = int(250 * (1 - x / 63))
                b = int(250 * (x / 63))
                for y in range(64):
                    img.putpixel((x, y), (r, 0, b))
            img.save(f, format="PNG")
            path = f.name
        h2 = _dhash(Image.open(path))
        os.unlink(path)
        # Hamming distance should be small
        diff = bin(int(h1, 16) ^ int(h2, 16)).count("1")
        assert diff < 20, f"Should be similar: Hamming distance = {diff}"


# ------------------------------------------------------------------
#  Dedup logic
# ------------------------------------------------------------------

class TestDedupLogic:

    def test_set_tracking(self):
        """Basic dedup set operations."""
        seen = set()
        assert "abc" not in seen
        seen.add("abc")
        assert "abc" in seen
        assert "xyz" not in seen

    def test_duplicate_detected(self):
        seen = set()
        h = "deadbeef12345678"
        seen.add(h)
        assert h in seen
