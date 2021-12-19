import { Flex, Text } from '@chakra-ui/react';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import BaseIndexPage from '../components/baseIndexPage';
import SelectInputWithFieldName from '../components/Inputs/SelectInputWithFieldName';
import TextInputWithFieldName from '../components/Inputs/textInputWithFieldName';
import { useSocket } from '../lib/socket';

enum OpponentType {
  Random = 'Random',
  Choose = 'Choose'
}

const BattleIndexPage = () => {
  const { socket } = useSocket();
  const [opponentType, setOpponentType] = useState<OpponentType>(
    OpponentType.Random
  );
  const [myAgent, setMyAgent] = useState<string>('');
  const [opponent, setOpponent] = useState<string>('');

  const emitRandomBattleStart = () => {
    socket?.emit('BattleStart', {
      agentUuid: myAgent,
      opponent: opponentType === OpponentType.Random ? 'Random' : opponent
    });
  };

  const handleOpponentTypeChange = e => {
    const newOpponentType = e.target.value;
    if (newOpponentType !== opponentType) {
      setOpponentType(newOpponentType);
    }
  };
  return (
    <>
      <BaseIndexPage>
        <TextInputWithFieldName
          label="My Agent Uuid"
          key="My Agent Uuid"
          id="My Agent Uuid"
          placeholder=""
          onChange={e => {
            setMyAgent(e.target.value);
          }}
        />
        <SelectInputWithFieldName
          id=""
          label="Oponent Type"
          onChange={handleOpponentTypeChange}
        >
          <option value="Random">Random</option>
          <option value="Choose">Choose</option>
        </SelectInputWithFieldName>
        {opponentType === OpponentType.Choose && (
          <TextInputWithFieldName
            label="Opponent Uuid"
            key="Opponent Uuid"
            id="Opponent Uuid"
            placeholder=""
            onChange={e => {
              setOpponent(e.target.value);
            }}
          />
        )}
      </BaseIndexPage>
      <Link to="/battle/game">
        <Flex alignItems="center" justifyContent="center" flexDir="column">
          <Flex
            alignItems="center"
            justifyContent="center"
            flexDir="column"
            w="500px"
            h="400px"
            backgroundColor="gray.100"
            p={5}
            borderRadius={20}
            borderColor="gray.200"
            borderWidth={3}
            _hover={{
              background: 'red.100',
              color: 'teal.500',
              cursor: 'pointer'
            }}
            onClick={emitRandomBattleStart}
          >
            <Text fontSize="60px">🔥Start Battle🔥</Text>
          </Flex>
        </Flex>
      </Link>
    </>
  );
};

export default BattleIndexPage;
