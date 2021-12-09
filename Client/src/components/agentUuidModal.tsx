import { useState } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  Button
} from '@chakra-ui/react';
import TextInputWithFieldName from './Inputs/textInputWithFieldName';

interface UuidModalProps {
  isOpen: boolean;
  onClose: () => void;
  handleGetModel: (uuid: string) => void;
}

const AgentUuidModal = ({
  isOpen,
  onClose,
  handleGetModel
}: UuidModalProps) => {
  const [uuid, setUuid] = useState('');
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Find Your Agent</ModalHeader>

        <ModalBody>
          <TextInputWithFieldName
            label="Agent id"
            id="agent_id"
            placeholder=""
            layout="1fr 2fr"
            onChange={e => {
              setUuid(e.target.value);
            }}
          />
        </ModalBody>

        <ModalFooter>
          <Button
            colorScheme="blue"
            mr={3}
            onClick={e => {
              handleGetModel(uuid);
            }}
          >
            Ok
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default AgentUuidModal;
