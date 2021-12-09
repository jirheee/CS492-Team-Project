import { Link } from 'react-router-dom';
import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';

const MainIndexPage = () => {
  return (
    <BaseIndexPage>
      <Link to="/agent">
        <HoverBtn
          baseText="Agent"
          hoverText="Create and Manage your own Agent"
        />
      </Link>
      <Link to="/battle">
        <HoverBtn baseText="Battle" hoverText="Battle your Agent" />
      </Link>
    </BaseIndexPage>
  );
};

export default MainIndexPage;
