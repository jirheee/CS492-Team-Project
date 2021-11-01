import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';

const AgentIndexPage = () => {
  return (
    <BaseIndexPage>
      <HoverBtn baseText="Create" hoverText="Create New Agent" />
      <HoverBtn baseText="Manage" hoverText="Train and Monitor your Agent" />
    </BaseIndexPage>
  );
};

export default AgentIndexPage;
