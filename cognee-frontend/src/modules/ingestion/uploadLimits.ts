// Backend caps concurrent processing per /v1/remember call — selecting more
// files than this in one batch risks partial failures under load.
export const MAX_FILES_PER_UPLOAD = 100;
