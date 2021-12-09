import { useState } from 'react';
import { Container, Flex, Spinner, Text } from '@chakra-ui/react';
import { CloseIcon, CheckIcon } from '@chakra-ui/icons';
import NNBoard from '../components/nnBoard';
import { HyperParameters } from '../model/types';
import AgentSummary from '../components/agentSummary';
import TrainSettingForm from '../components/trainSettingForm';
import AgentUuidModal from '../components/agentUuidModal';
import customAxios from '../lib/api/request';
import Model from '../model/model';
import LineChart from '../components/lineChart';

export enum TrainStatus {
  NOT_TRAINED = 'Not Trained',
  TRAINING = 'Training',
  TRAIN_FINISHED = 'Train Finished'
}

const dummyValidationLossHistory = new Array(50)
  .fill(0)
  .map(_ => Math.random() * 100);
const dummyTrainLossHistory = new Array(50)
  .fill(0)
  .map(_ => Math.random() * 100);
const dummyWinRateHistory = new Array(50).fill(0).map(_ => Math.random() * 100);

const AgentManagementPage = () => {
  const [model, setModel] = useState<Model | undefined>(undefined);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [trainInfo, setTrainInfo] = useState<
    | {
        hyperparameters: HyperParameters;
        trainStatus: TrainStatus;
      }
    | undefined
  >();
  const [trainTrainLossHistory, setTrainLossHistory] = useState<number[]>(
    dummyTrainLossHistory
  );
  const [trainValidationLossHistory, setTrainValidationLossHistory] = useState<
    number[]
  >(dummyValidationLossHistory);
  const [trainWinRateHistory, setTrainWinRateHistory] =
    useState<number[]>(dummyWinRateHistory);
  const [agentUuid, setAgentUuid] = useState('');

  const handleGetModel = (uuid: string) => {
    setAgentUuid(uuid);
    customAxios.get(`/agent/${uuid}`).then(res => {
      const {
        data: { model, trainInfo, status }
      }: {
        data: {
          model: Model;
          trainInfo: {
            hyperparameters: HyperParameters;
            trainStatus: TrainStatus;
          };
          status: number;
        };
      } = res;
      if (status === 200) {
        setModel(model);
        setModelLoaded(true);
        setTrainInfo(trainInfo);
      }
    });
  };

  const empty = () => {};

  const handleTrainSubmit = e => {
    e.preventDefault();
    const { target } = e;
    const hyperparameters: HyperParameters = {
      lr: +target.lr.value,
      buffer_size: +target.buffer_size.value,
      batch_size: +target.batch_size.value,
      epochs: +target.epochs.value
    };
    customAxios
      .post(
        '/agent/train',
        { hyperparameters, agentUuid },
        {
          headers: { 'Content-Type': `application/json` }
        }
      )
      .then(res => {
        const {
          data: { trainInfo }
        } = res;
        setTrainInfo(trainInfo);
      });
  };

  return (
    <Flex
      flexDir="column"
      justifyContent="center"
      backgroundColor="gray.50"
      h="100vh"
      overflowY="scroll"
    >
      <AgentUuidModal
        isOpen={!modelLoaded}
        onClose={empty}
        handleGetModel={handleGetModel}
      />
      <Container maxW={1200} backgroundColor="white" h="full" p={5}>
        <AgentSummary model={model} trainInfo={trainInfo} />
        <Text fontSize="lg" fontWeight="bold" mb={3} mt={3}>
          Agent Network Architecture
        </Text>
        <NNBoard layers={model?.layers ?? []} isModifiable={false} h="400px" />
        <Flex flexDir="column" mt={3} mb={5}>
          <Text fontSize="lg" fontWeight="bold" mb={3}>
            Train Information
          </Text>
          <Flex alignContent="center">
            <Text fontSize="md" fontWeight="bold" pr={3}>
              Train Status: {trainInfo?.trainStatus ?? 'NaN'}
            </Text>
            {trainInfo?.trainStatus === TrainStatus.TRAINING && <Spinner />}
            {trainInfo?.trainStatus === TrainStatus.NOT_TRAINED && (
              <CloseIcon p={0.5} color="red.500" />
            )}
            {trainInfo?.trainStatus === TrainStatus.TRAIN_FINISHED && (
              <CheckIcon p={0.5} color="green.500" />
            )}
          </Flex>

          {trainInfo && (
            <Flex flexDir="column" pb={10}>
              {trainInfo?.trainStatus === TrainStatus.NOT_TRAINED && (
                <TrainSettingForm onSubmit={handleTrainSubmit} />
              )}
              {trainInfo?.trainStatus !== TrainStatus.NOT_TRAINED && (
                <>
                  <LineChart
                    dataPointsSet={[
                      trainTrainLossHistory,
                      trainValidationLossHistory
                    ]}
                    title="Train Loss History"
                    labels={trainTrainLossHistory.map((_, i) => i)}
                    datasetNameSet={['Train Loss', 'Validation Loss']}
                    borderColorSet={['rgb(255, 99, 132)', 'rgb(53, 162, 235)']}
                    backgroundColorSet={[
                      'rgba(255, 99, 132, 0.5)',
                      'rgba(53, 162, 235, 0.5)'
                    ]}
                  />
                  <LineChart
                    dataPointsSet={[trainWinRateHistory]}
                    title="Train Win Rate History"
                    labels={trainWinRateHistory.map((_, i) => i)}
                    datasetNameSet={[`${model?.name} Train Win Rate History`]}
                    borderColorSet={['rgb(255, 99, 132)']}
                    backgroundColorSet={['rgba(255, 99, 132, 0.5)']}
                  />
                </>
              )}
            </Flex>
          )}
        </Flex>
      </Container>
    </Flex>
  );
};

export default AgentManagementPage;
