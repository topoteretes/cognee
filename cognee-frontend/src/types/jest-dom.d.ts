// The jest-dom matcher typings reach the program only through jest.setup.tsx,
// which the public-repo sync does not copy, so every synced test using
// toHaveTextContent or toBeInTheDocument failed `tsc --noEmit` there and broke
// the weekly sync job. Importing the augmentation from a .d.ts inside src/ puts
// it in the program both here and in the mirror, for every test file at once.
import "@testing-library/jest-dom";
