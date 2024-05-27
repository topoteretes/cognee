import { withStyles } from 'ohmy-ui';
import styles from './WizardContent.module.css';

const WizardContent = withStyles<{ children: React.ReactNode }>('div', { className: styles.wizardContent });

export default WizardContent;
