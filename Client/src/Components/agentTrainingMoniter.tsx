import { useState } from 'react';

enum TrainStatus {
  UNTRAINED = 'Not Trained',
  TRAINING = 'Training',
  FINISHED = 'Finished'
}

const AgentTrainingMonitor = () => {
  const [trainStatus, setTrainStatus] = useState<TrainStatus>(
    TrainStatus.UNTRAINED
  );

  return;
};

export default AgentTrainingMonitor;
