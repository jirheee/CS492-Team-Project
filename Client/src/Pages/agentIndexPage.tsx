import { Link } from 'react-router-dom';
import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';

const AgentIndexPage = () => {
  return (
    <BaseIndexPage>
      <Link to="/agent/create">
        <HoverBtn baseText="Create" hoverText="Create New Agent" />
      </Link>
      <Link to="/agent/manage">
        <HoverBtn baseText="Manage" hoverText="Train and Monitor your Agent" />
      </Link>
    </BaseIndexPage>
  );
};

export default AgentIndexPage;
